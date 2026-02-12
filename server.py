"""
AlphaGenome Web Demo - Python Backend
Wraps the AlphaGenome gRPC API for the web frontend.
"""

import json
import os
import traceback
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY = os.environ.get("ALPHAGENOME_API_KEY", "")

# Output type string -> attribute name on Output object
OUTPUT_ATTR = {
    "DNASE": "dnase",
    "ATAC": "atac",
    "CAGE": "cage",
    "RNA_SEQ": "rna_seq",
    "CHIP_HISTONE": "chip_histone",
    "CHIP_TF": "chip_tf",
    "PROCAP": "procap",
    "SPLICE_SITES": "splice_sites",
    "SPLICE_SITE_USAGE": "splice_site_usage",
    "SPLICE_JUNCTIONS": "splice_junctions",
    "CONTACT_MAPS": "contact_maps",
}

# Global model (lazy init; gRPC channel is expensive)
_model = None


def get_model():
    global _model
    if not API_KEY:
        return None
    if _model is None:
        print("Creating AlphaGenome gRPC client...", flush=True)
        _model = dna_client.create(API_KEY, timeout=60)
        print("Client ready.", flush=True)
    return _model


def nearest_length(length):
    """Snap to the smallest supported sequence length >= length."""
    for s in [
        dna_client.SEQUENCE_LENGTH_16KB,
        dna_client.SEQUENCE_LENGTH_100KB,
        dna_client.SEQUENCE_LENGTH_500KB,
        dna_client.SEQUENCE_LENGTH_1MB,
    ]:
        if length <= s:
            return s
    return dna_client.SEQUENCE_LENGTH_1MB


def downsample(arr, max_points=2000):
    """Downsample a 1-D numpy array by averaging bins."""
    if len(arr) <= max_points:
        return arr.tolist()
    step = len(arr) / max_points
    result = []
    for i in range(max_points):
        lo = int(i * step)
        hi = int((i + 1) * step)
        result.append(float(np.mean(arr[lo:hi])))
    return result


def get_track_data(output, output_type):
    """Extract TrackData from an Output object by output_type string."""
    attr = OUTPUT_ATTR.get(output_type)
    if attr is None:
        raise ValueError(f"Unknown output type: {output_type}")
    td = getattr(output, attr, None)
    if td is None:
        raise ValueError(f"No data for {output_type}; model returned None.")
    return td


# ──────────────── Routes ────────────────


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/status")
def api_status():
    mode = "live" if API_KEY else "demo"
    return jsonify({"mode": mode, "api_key_set": bool(API_KEY)})


@app.route("/api/predict/sequence", methods=["POST"])
def predict_sequence():
    data = request.json
    sequence = data.get("sequence", "").upper().strip()
    output_type = data.get("output_type", "DNASE")
    ontology_term = data.get("ontology_term", "UBERON:0002048")

    if not sequence:
        return jsonify({"error": "Empty sequence"}), 400

    try:
        model = get_model()
        otype = getattr(dna_client.OutputType, output_type)
        seq_len = nearest_length(len(sequence))
        padded = sequence.center(seq_len, "N")

        output = model.predict_sequence(
            sequence=padded,
            requested_outputs=[otype],
            ontology_terms=[ontology_term],
        )

        td = get_track_data(output, output_type)
        # Average across tracks if multiple, then downsample
        signal = td.values.mean(axis=1) if td.values.ndim > 1 else td.values
        values = downsample(signal, max_points=2000)
        metadata = td.metadata.to_dict(orient="records")

        return jsonify({
            "mode": "live",
            "data": {
                "type": "track",
                "values": values,
                "output_type": output_type,
                "sequence_length": seq_len,
                "n_tracks": int(td.values.shape[1]) if td.values.ndim > 1 else 1,
                "metadata": metadata,
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/interval", methods=["POST"])
def predict_interval():
    data = request.json
    chrom = data.get("chromosome", "chr22")
    start = int(data.get("start", 35677410))
    end = int(data.get("end", 36725986))
    output_type = data.get("output_type", "RNA_SEQ")
    ontology_term = data.get("ontology_term", "UBERON:0001157")

    try:
        model = get_model()
        otype = getattr(dna_client.OutputType, output_type)
        interval = genome.Interval(chrom, start, end)
        seq_len = nearest_length(end - start)
        interval = interval.resize(seq_len)

        output = model.predict_interval(
            interval=interval,
            requested_outputs=[otype],
            ontology_terms=[ontology_term],
        )

        td = get_track_data(output, output_type)

        # Contact maps are 2-D matrices
        if output_type == "CONTACT_MAPS":
            mat = td.values
            # Downsample to max 200x200
            if mat.shape[0] > 200:
                step = mat.shape[0] // 200
                mat = mat[::step, ::step]
            return jsonify({
                "mode": "live",
                "data": {
                    "type": "contact_map",
                    "matrix": mat.tolist(),
                    "size": int(mat.shape[0]),
                    "interval": {"chromosome": chrom, "start": start, "end": end},
                },
            })

        signal = td.values.mean(axis=1) if td.values.ndim > 1 else td.values
        values = downsample(signal, max_points=2000)
        metadata = td.metadata.to_dict(orient="records")

        return jsonify({
            "mode": "live",
            "data": {
                "type": "track",
                "values": values,
                "output_type": output_type,
                "interval": {"chromosome": chrom, "start": start, "end": end},
                "n_tracks": int(td.values.shape[1]) if td.values.ndim > 1 else 1,
                "metadata": metadata,
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/variant", methods=["POST"])
def predict_variant():
    data = request.json
    chrom = data.get("chromosome", "chr22")
    position = int(data.get("position", 36201698))
    ref = data.get("ref", "A")
    alt = data.get("alt", "C")
    output_type = data.get("output_type", "RNA_SEQ")
    ontology_term = data.get("ontology_term", "UBERON:0001157")

    try:
        model = get_model()
        otype = getattr(dna_client.OutputType, output_type)

        variant = genome.Variant(
            chromosome=chrom,
            position=position,
            reference_bases=ref,
            alternate_bases=alt,
        )
        interval = genome.Interval(chrom, position - 500000, position + 500000)
        interval = interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

        # Predict variant (ref vs alt)
        var_output = model.predict_variant(
            interval=interval,
            variant=variant,
            requested_outputs=[otype],
            ontology_terms=[ontology_term],
        )

        ref_td = get_track_data(var_output.reference, output_type)
        alt_td = get_track_data(var_output.alternate, output_type)

        ref_signal = ref_td.values.mean(axis=1) if ref_td.values.ndim > 1 else ref_td.values
        alt_signal = alt_td.values.mean(axis=1) if alt_td.values.ndim > 1 else alt_td.values

        ref_values = downsample(ref_signal, max_points=1000)
        alt_values = downsample(alt_signal, max_points=1000)

        # Score the variant
        scores = {}
        scorer_key = output_type
        if scorer_key in variant_scorers.RECOMMENDED_VARIANT_SCORERS:
            scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS[scorer_key]
            score_results = model.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=[scorer],
            )
            if score_results:
                adata = score_results[0]
                # adata.X is (n_genes x n_tracks), adata.layers['quantiles'] too
                raw_scores = adata.X
                max_abs = float(np.max(np.abs(raw_scores)))
                mean_abs = float(np.mean(np.abs(raw_scores)))

                quantiles = adata.layers.get("quantiles")
                max_quantile = float(np.max(quantiles)) if quantiles is not None else None

                # Top affected genes
                gene_effects = np.abs(raw_scores).max(axis=1)
                top_idx = np.argsort(gene_effects)[::-1][:5]
                top_genes = []
                for idx in top_idx:
                    gene_name = adata.obs.iloc[idx].get("gene_name", f"gene_{idx}")
                    top_genes.append({
                        "name": str(gene_name),
                        "score": round(float(gene_effects[idx]), 4),
                    })

                scores = {
                    "max_abs_score": round(max_abs, 4),
                    "mean_abs_score": round(mean_abs, 4),
                    "max_quantile": round(max_quantile, 4) if max_quantile else None,
                    "n_genes": int(adata.n_obs),
                    "n_tracks": int(adata.n_vars),
                    "top_genes": top_genes,
                }

        return jsonify({
            "mode": "live",
            "data": {
                "type": "variant_comparison",
                "reference": ref_values,
                "alternate": alt_values,
                "scores": scores,
                "variant": {
                    "chromosome": chrom,
                    "position": position,
                    "ref": ref,
                    "alt": alt,
                },
                "output_type": output_type,
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n=== AlphaGenome Web Demo ===")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print("Warming up gRPC client (first request may take a moment)...")
    print("Open http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
