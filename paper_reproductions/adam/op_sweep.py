"""Sweep tau_novelty and tau_conf to find an operating point."""
from __future__ import annotations

import pandas as pd

from data_loader import default_splits
import pipeline as dp


def main() -> None:
    train, test = default_splits()
    prof = dp.mine_profile(train, s_min=0.005)
    print(f"profile size1={len(prof.size1)} size2={len(prof.size2)}")

    rows = []
    m_train_base = dp.run_miner(train, prof, tau_novelty=0.0)
    m_test_base = dp.run_miner(test, prof, tau_novelty=0.0)

    for tau_n in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70):
        flagged_train = m_train_base.row_novelty > tau_n
        flagged_test = m_test_base.row_novelty > tau_n
        tc = dp.train_clf(train.df, train.y,
                           m_train_base.row_novelty, m_train_base.win_novelty)
        pred, conf = dp.predict_clf(tc, test.df,
                                      m_test_base.row_novelty,
                                      m_test_base.win_novelty)
        for tau_c in (0.50, 0.60, 0.70, 0.80, 0.90):
            r = dp.evaluate(test, flagged_test, pred, conf, tau_conf=tau_c)
            rows.append({
                "tau_novelty": tau_n, "tau_conf": tau_c,
                "FAR": r.false_alarm_rate,
                "inst_det_overall": r.instances_detected / max(1, r.instances_total),
                "inst_det_DoS": r.instance_detection_rate["DoS"],
                "inst_det_Probe": r.instance_detection_rate["Probe"],
                "inst_det_R2L": r.instance_detection_rate["R2L"],
                "inst_det_U2R": r.instance_detection_rate["U2R"],
                "inst_id_acc": r.instance_identification_accuracy,
                "conn_recall": r.recall,
                "conn_precision": r.precision,
            })
    df = pd.DataFrame(rows)
    df.to_csv("out/op_sweep.csv", index=False)
    with pd.option_context("display.precision", 3, "display.width", 160,
                            "display.max_columns", 20):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
