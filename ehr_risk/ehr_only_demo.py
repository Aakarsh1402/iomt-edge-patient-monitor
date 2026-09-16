"""EHR risk engine - standalone demo.

Scores the sample patients in EHR_patient_records.json with the trained
multi-task model and prints the headline risks.

    python ehr_only_demo.py               # interactive: switch patient / deep dive / quit
    python ehr_only_demo.py --all         # score every patient and exit (non-interactive)
    python ehr_only_demo.py --patient 102 --top 10
"""
import argparse
import json
import os
import sys

from ehr_engine import HERE, EHRRiskEngine
from ehr_vectorize import map_history, pretty_target, top_risks

PATIENT_DB = os.path.join(HERE, "EHR_patient_records.json")


def print_summary(record, result):
    key = result.key_risks()
    print(f"   Name: {record['name']}   (age {record['age']}, {record['gender']})")
    print(f"   Inference time: {result.latency_ms:.1f} ms for {result.n_rows} observation rows")
    print("-" * 40)
    print(f"   Fall risk:     {key['fall'] * 100:5.1f}%")
    print(f"   Seizure risk:  {key['seizure'] * 100:5.1f}%")
    print(f"   Cardiac risk:  {key['cardiac'] * 100:5.1f}%")
    print(f"   Sepsis risk:   {key['sepsis'] * 100:5.1f}%")
    print(f"   Stroke risk:   {key['stroke'] * 100:5.1f}%")
    print("-" * 40)
    for w in result.warnings:
        print(f"   ! {w}")


def print_deep_dive(engine, record, result, top_n):
    risks = result.risks
    print("\n" + "=" * 50)
    print("NEURAL NETWORK INTERNALS")
    print("=" * 50)
    active = result.active_features
    flags = sorted(k for k in active if k.startswith("history_") and k not in engine.unsupported)
    numeric = {k: v for k, v in active.items() if not k.startswith("history_")}
    print(f"INPUT LAYER ({len(active)} of {len(engine.features)} features non-zero, "
          f"spread over {result.n_rows} observation rows):")
    print(f"   History flags set: {', '.join(f.replace('history_', '') for f in flags) or 'none'}")
    print(f"   Numeric inputs:    {', '.join(f'{k}={v:g}' for k, v in numeric.items())}")
    mapping = map_history(record.get("history", []), engine.features)
    unmapped = [h for h, f in mapping.items() if f is None or f in engine.unsupported]
    if unmapped:
        print(f"   (no usable model feature for: {', '.join(unmapped)})")

    print(f"\nOUTPUT LAYER (top {top_n} of {len(engine.targets)} predicted risks):")
    for i, (name, prob) in enumerate(top_risks(risks.values(), list(risks), top_n), 1):
        bar = "#" * int(prob * 20)
        print(f"   {i:>2}. {pretty_target(name):<26} |{bar:<20}| {prob * 100:6.2f}%")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=PATIENT_DB, help="patient records JSON")
    parser.add_argument("--patient", help="patient id to start with (default: first in file)")
    parser.add_argument("--all", action="store_true", help="score all patients and exit")
    parser.add_argument("--top", type=int, default=5, help="rows in the deep-dive risk table")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EHR RISK ENGINE: STANDALONE DEMO")
    print("=" * 60)

    try:
        with open(args.db) as f:
            patient_db = json.load(f)
    except OSError as e:
        sys.exit(f"Could not read patient database: {e}")
    if not patient_db:
        sys.exit(f"{args.db} contains no patients")

    print("[INIT] Loading Keras model & scaler...")
    try:
        engine = EHRRiskEngine()
    except Exception as e:  # missing artifacts, version mismatch, ...
        sys.exit(f"ERROR: could not load model artifacts from {HERE}: {e}")
    print(f"      Model loaded ({len(engine.features)} inputs -> {len(engine.targets)} outputs)")

    patient_ids = list(patient_db)
    if args.patient and args.patient not in patient_db:
        sys.exit(f"Patient {args.patient!r} not in {args.db}; known ids: {', '.join(patient_ids)}")
    current = args.patient or patient_ids[0]

    def analyze(pid):
        record = patient_db[pid]
        print(f"\nPATIENT {pid}")
        result = engine.score(record)
        print_summary(record, result)
        return record, result

    if args.all:
        for pid in patient_ids:
            record, result = analyze(pid)
            print_deep_dive(engine, record, result, args.top)
        return

    record, result = analyze(current)
    while True:
        print("\n[S]witch patient | [D]eep dive | [Q]uit")
        try:
            cmd = input("Command > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd == "s":
            current = patient_ids[(patient_ids.index(current) + 1) % len(patient_ids)]
            record, result = analyze(current)
        elif cmd == "d":
            print_deep_dive(engine, record, result, args.top)
        elif cmd == "q":
            break


if __name__ == "__main__":
    main()
