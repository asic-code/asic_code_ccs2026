"""ADAM runner (DARPA'98)."""
from __future__ import annotations

import sys


def cmd_single() -> None:
    from data_loader import default_splits
    from pipeline import run_end_to_end, format_result
    train, test = default_splits()
    print(format_result(run_end_to_end(train, test)))


def cmd_audit() -> None:
    from audit_isolation import main as audit_main
    audit_main()


def cmd_removal() -> None:
    from dataset_lost import main as ablation_main
    ablation_main()


def cmd_sweep() -> None:
    from op_sweep import main as sweep_main
    sweep_main()


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "single"
    {"single": cmd_single, "audit": cmd_audit,
     "removal": cmd_removal, "sweep": cmd_sweep}[cmd]()


if __name__ == "__main__":
    main()
