"""Benchmark the TFLite export of the EHR risk model and check it agrees with Keras.

    python test_ehr_performance.py                 # latency + top risks for patient 101
    python test_ehr_performance.py --patient 102 --runs 500 --no-compare

Requires ehr_tflite_outputs.json (run ehr_tflite_map.py once after training):
TFLite does not preserve the order of the 51 output heads, so reading them by
index - as this script used to - labels every probability with the wrong risk.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from ehr_tflite_map import MAP_FILE, load_mapping, make_interpreter  # noqa: E402
from ehr_vectorize import pretty_target  # noqa: E402


def score_tflite(interpreter, mapping, x_scaled):
    """Run each row of x_scaled; return ({target: max prob over rows}, per-row latencies ms)."""
    inp = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    best, latencies = {}, []
    for row in x_scaled:
        interpreter.set_tensor(inp["index"], row[None, :].astype(np.float32))
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)
        for d in outputs:
            target = mapping[d["index"]]
            val = float(interpreter.get_tensor(d["index"]).reshape(-1)[0])
            best[target] = max(best.get(target, 0.0), val)
    return best, latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", default="101")
    parser.add_argument("--runs", type=int, default=100, help="timed inference runs")
    parser.add_argument("--no-compare", action="store_true", help="skip loading the Keras model for comparison")
    args = parser.parse_args()

    if not os.path.exists(MAP_FILE):
        sys.exit(f"{MAP_FILE} not found. Run `python ehr_tflite_map.py` first to recover the output order.")
    mapping = load_mapping()

    # Keras engine gives us the training-shaped observation rows and, optionally, reference scores.
    from ehr_engine import EHRRiskEngine
    engine = EHRRiskEngine()
    with open(os.path.join(HERE, "EHR_patient_records.json")) as f:
        record = json.load(f)[args.patient]
    keras_result = engine.score(record)

    from ehr_vectorize import build_feature_vector
    rows = [build_feature_vector(row, engine.features)[0]
            for label, row in engine._observation_rows(record)
            if label == "demographics+history" or label in engine.features]
    x = engine.scaler.transform(np.concatenate(rows)).astype(np.float32)
    x[:, engine._unsupported_idx] = 0.0

    interpreter = make_interpreter(os.path.join(HERE, "ehr_model.tflite"))
    print(f"TFLite model loaded; input shape {interpreter.get_input_details()[0]['shape']}, "
          f"{len(mapping)} outputs mapped\n")

    print(f"--- Warm-up on {len(x)} observation rows for patient {args.patient} ({record['name']}) ---")
    score_tflite(interpreter, mapping, x)

    print(f"--- Timing {args.runs} single-row inferences ---")
    single = x[:1]
    latencies = []
    for _ in range(args.runs):
        _, lat = score_tflite(interpreter, mapping, single)
        latencies += lat
    print(f"TFLite latency: mean {np.mean(latencies):.3f} ms, p95 {np.percentile(latencies, 95):.3f} ms per row\n")

    risks, _ = score_tflite(interpreter, mapping, x)
    print("--- Top 5 risks (TFLite) ---")
    for name, prob in sorted(risks.items(), key=lambda kv: -kv[1])[:5]:
        line = f"{pretty_target(name):<26} {prob:7.2%}"
        if not args.no_compare:
            line += f"   Keras {keras_result.risks[name]:7.2%}"
        print(line)

    if not args.no_compare:
        diffs = [abs(risks[t] - keras_result.risks[t]) for t in engine.targets]
        print(f"\nMax |TFLite - Keras| over all {len(diffs)} heads: {max(diffs):.4f} "
              "(drift is expected from TFLite weight quantisation; labels are what matter)")
        top = lambda d: {k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])[:5]}  # noqa: E731
        if top(risks) != top(keras_result.risks):
            sys.exit("TFLite and Keras disagree on the top-5 risks - the output mapping is stale; "
                     "re-run ehr_tflite_map.py")
        print("TFLite output labels agree with Keras (same top-5 risks).")


if __name__ == "__main__":
    main()
