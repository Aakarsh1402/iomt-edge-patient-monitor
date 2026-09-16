"""Work out which TFLite output tensor corresponds to which risk head.

The multi-task model has 51 named sigmoid heads. TFLite conversion does not
preserve their order, and the converted tensor names (StatefulPartitionedCall
_1:N) do not encode it either, so reading outputs by index - as the benchmark
script used to - attaches every probability to the wrong risk name.

This script runs the Keras model and the TFLite model on the same random
inputs, matches each TFLite output to the Keras head it reproduces, verifies
the result is a clean one-to-one mapping, and writes it to
ehr_tflite_outputs.json as {tflite_tensor_index: target_name}.

    python ehr_tflite_map.py            # after ehr_train.py has written ehr_model.tflite
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(HERE, "ehr_tflite_outputs.json")
N_PROBES = 512
ACTIVE_STD = 1e-2        # a head whose output varies less than this over the probes is effectively constant
MAX_ERROR = 0.05         # TFLite's weight quantisation shifts outputs by ~0.01; a true match stays under this
MIN_CORRELATION = 0.9    # ...and must rise and fall with the Keras head


def make_interpreter(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_outputs(interpreter, x):
    """Run one (1, n_features) input; return {tensor_index: scalar}."""
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp["index"], x.astype(np.float32))
    interpreter.invoke()
    return {d["index"]: float(interpreter.get_tensor(d["index"]).reshape(-1)[0])
            for d in interpreter.get_output_details()}


def probe_inputs(features, scaler, n=512, seed=0):
    """Inputs shaped like training rows, spanning the clinical regimes the
    heads were trained on.

    Random z-space noise never produces "history_seizure=1 and creatinine>1.5",
    so most heads look constant under it. Each probe is therefore a synthetic
    training-style row: age, gender, every history flag a coin toss, and at
    most ONE vital/lab (training rows hold a single observation each) drawn
    from a wide clinical range. Rows are scaled exactly like real records and
    features that were constant in training are pinned to 0 in z-space.
    """
    rng = np.random.default_rng(seed)
    n_features = len(features)
    numeric = [j for j, name in enumerate(features)
               if name not in ("age", "gender") and not name.startswith("history_")]
    raw = np.zeros((n, n_features), dtype=np.float32)
    for j, name in enumerate(features):
        if name == "age":
            raw[:, j] = rng.uniform(18, 95, n)
        elif name == "gender" or name.startswith("history_"):
            raw[:, j] = rng.integers(0, 2, n)
    for i in range(n):
        if rng.random() < 0.8:  # 20% of rows: demographics + history only
            j = numeric[rng.integers(len(numeric))]
            # Training rows are mostly 0 for any given lab, so the scaler's
            # mean/sd understate the clinical range; stretch generously.
            raw[i, j] = rng.uniform(0, scaler.mean_[j] + 15 * scaler.scale_[j])
    z = scaler.transform(raw).astype(np.float32)
    z[:, scaler.var_ == 0] = 0.0
    # A block of plain z-space noise as well, for heads driven by scale alone.
    noise = rng.standard_normal((n // 4, n_features)).astype(np.float32) * 3
    return np.concatenate([z, noise])


def derive_mapping(keras_predict, interpreter, targets, features, scaler, n_probes=N_PROBES):
    """keras_predict: (n, n_features) -> (n, n_targets) in `targets` order.

    Returns (mapping {tensor_index: target}, constant_targets). Every TFLite
    output that moves over the probes is matched to the Keras head that
    reproduces it. Outputs that never move are numerically indistinguishable
    from each other, so they are paired off with the unmatched heads by
    elimination and reported so the caller knows which labels are nominal.
    """
    X = probe_inputs(features, scaler, n_probes)
    K = keras_predict(X)                                  # (n_probes, n_targets)
    T = {}                                                # tensor_index -> (n_probes,)
    for i in range(len(X)):
        for idx, val in tflite_outputs(interpreter, X[i:i + 1]).items():
            T.setdefault(idx, []).append(val)
    T = {idx: np.array(v) for idx, v in T.items()}
    if len(T) != len(targets):
        raise RuntimeError(f"TFLite model has {len(T)} outputs but {len(targets)} targets are listed")

    # One-to-one assignment between the outputs and heads that actually move,
    # minimising total mean-absolute-error; then verify every pair.
    active_outputs = [idx for idx in sorted(T) if T[idx].std() > ACTIVE_STD]
    active_heads = [j for j in range(len(targets)) if K[:, j].std() > ACTIVE_STD]
    if len(active_outputs) != len(active_heads):
        raise RuntimeError(f"{len(active_heads)} Keras heads move over the probes but "
                           f"{len(active_outputs)} TFLite outputs do; try more probes")
    cost = np.array([[np.abs(K[:, j] - T[idx]).mean() for j in active_heads] for idx in active_outputs])
    rows, cols = linear_sum_assignment(cost)
    mapping, used = {}, set()
    for r, c in zip(rows, cols):
        idx, head = active_outputs[r], active_heads[c]
        corr = np.corrcoef(K[:, head], T[idx])[0, 1]
        if cost[r, c] > MAX_ERROR or corr < MIN_CORRELATION:
            raise RuntimeError(f"TFLite output {idx} -> {targets[head]} is a poor match "
                               f"(err={cost[r, c]:.4f}, r={corr:.3f})")
        used.add(head)
        mapping[idx] = targets[head]

    leftover_heads = [j for j in range(len(targets)) if j not in used]
    leftover_outputs = [idx for idx in sorted(T) if idx not in mapping]
    for idx, head in zip(leftover_outputs, leftover_heads):
        mapping[idx] = targets[head]
    return mapping, [targets[j] for j in leftover_heads]


def load_mapping(map_file=MAP_FILE):
    """{tflite_tensor_index: target_name}. Raises FileNotFoundError if ehr_tflite_map.py has not been run."""
    with open(map_file) as f:
        return {int(k): v for k, v in json.load(f)["outputs"].items()}


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    sys.path.insert(0, HERE)
    from ehr_engine import EHRRiskEngine

    engine = EHRRiskEngine()
    interpreter = make_interpreter(os.path.join(HERE, "ehr_model.tflite"))
    mapping, constant = derive_mapping(engine._predict, interpreter, engine.targets, engine.features, engine.scaler)

    with open(MAP_FILE, "w") as f:
        json.dump({"outputs": {str(k): v for k, v in sorted(mapping.items())},
                   "constant_heads": constant}, f, indent=2)
    output_order = [interpreter.get_output_details()[i]["index"] for i in range(len(mapping))]
    shuffled = sum(1 for i, idx in enumerate(output_order) if engine.targets[i] != mapping[idx])
    print(f"Mapped {len(mapping)} TFLite outputs -> {MAP_FILE}")
    print(f"{shuffled} of {len(mapping)} outputs are NOT in Keras head order; index-based reads would mislabel them.")
    print(f"{len(constant)} heads are constant functions (label never fired in training): "
          + ", ".join(t.replace("risk_", "").replace("_24h", "") for t in constant))


if __name__ == "__main__":
    main()
