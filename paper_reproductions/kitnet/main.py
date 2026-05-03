"""CLI dispatcher for the KitNET / Mirai reproduction."""
from __future__ import annotations
import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    sub.add_parser("audit")
    sub.add_parser("phase3")
    sub.add_parser("phase4")

    args, rest = ap.parse_known_args()
    if args.cmd == "sanity":
        from data_loader import load_mirai, sanity_print
        sanity_print(load_mirai(fraction=1.0, seed=0))
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
