"""Run every check in the project and print a scoreboard.

    python run_checks.py            # unit tests + every module's evaluation + the demos
    python run_checks.py --quick    # unit tests and the fast demos only

Each step is an ordinary command you can run by hand (shown in the output).
Steps that need an optional dependency you have not installed are reported
as SKIPPED rather than failing, so a PyTorch-only or TensorFlow-only machine
still gets a meaningful result. Exit status is 1 if anything FAILED.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def have(module):
    return importlib.util.find_spec(module) is not None


def steps(quick):
    tf, torch = have("tensorflow"), have("torch")
    yield "unit tests", [PY, "-m", "pytest", "tests", "-q", "--no-header", "-p", "no:cacheprovider"], True
    yield "IMU: sanity check on generated motion", [PY, "gesture_recognition/sanity_check.py"], torch
    if not quick:
        yield "IMU: unseen-subject evaluation", [PY, "gesture_recognition/evaluate_model.py"], torch
        yield "ECG: unseen-patient evaluation (MIT-BIH DS2)", [PY, "arrhythmia/ecg_evaluate.py"], tf
    yield "ECG: score the collected sensor recordings", [PY, "arrhythmia/ecg_infer.py", "arrhythmia/raw_data/collected"], tf
    yield "EHR: risk scores for every sample patient", [PY, "ehr_risk/ehr_only_demo.py", "--all"], tf
    yield "EHR: TFLite export agrees with Keras", [PY, "ehr_risk/test_ehr_performance.py"], tf
    yield "Vision: checkpoint loads (no camera needed)", [PY, "-c",
          "import sys; sys.path.insert(0, 'vision'); from vision_model import load_model; "
          "m, c = load_model(); print('classes', c)"], torch
    yield "Fusion: scripted ward scenario", [PY, "fusion/run_monitor.py", "--delay", "0", "--no-color"], torch
    yield "Fusion: live MQTT pipeline (broker + node + monitor)", \
        [PY, "fusion/live_demo.py", "--speed", "8", "--no-color"], torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="stream each step's output")
    args = parser.parse_args()

    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONIOENCODING": "utf-8"}
    results = []
    for name, cmd, available in steps(args.quick):
        print(f"\n=== {name}\n    $ {' '.join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd)}")
        if not available:
            print("    SKIPPED (optional dependency not installed)")
            results.append((name, "SKIPPED", 0.0))
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                              stdout=None if args.verbose else subprocess.PIPE, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        status = "PASSED" if proc.returncode == 0 else "FAILED"
        if not args.verbose:
            tail = (proc.stdout or "").strip().splitlines()[-8:]
            for line in tail:
                print("    | " + line)
        print(f"    {status} in {dt:.1f}s")
        results.append((name, status, dt))

    print("\n" + "=" * 70)
    for name, status, dt in results:
        print(f"{status:<8}{dt:6.1f}s  {name}")
    failed = sum(s == "FAILED" for _, s, _ in results)
    print("=" * 70)
    print(f"{len(results) - failed}/{len(results)} steps ok" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
