"""CLI dispatcher for the Mateen / Kitsune reproduction."""
from __future__ import annotations
import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity", help="load + print sanity for the full split")
    sub.add_parser("audit", help="run audit_isolation")
    sub.add_parser("phase3", help="single end-to-end run")
    sub.add_parser("phase4", help="dataset-lost ablation")

    args, rest = ap.parse_known_args()
    if args.cmd == "sanity":
        from data_loader import load_kitsune, sanity_print
        s = load_kitsune(fraction=1.0, seed=0)
        sanity_print(s)
    elif args.cmd == "audit":
        sys.argv = ["audit_isolation.py", *rest]
        import audit_isolation as m
        m.main()
    elif args.cmd == "phase3":
        sys.argv = ["evaluate.py", *rest]
        import evaluate as m
        m.main()
    elif args.cmd == "phase4":
        sys.argv = ["dataset_lost.py", *rest]
        import dataset_lost as m
        m.main()


if __name__ == "__main__":
    main()
