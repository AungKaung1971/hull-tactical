from __future__ import annotations
import argparse

from .strategy import SizingConfig
from .submit import submit_allocations

def main():
    ap = argparse.ArgumentParser(description="Create Hull Tactical submission.csv")
    ap.add_argument("--train", required=True, help="Path to train.csv")
    ap.add_argument("--test", required=True, help="Path to test.csv")
    ap.add_argument("--preds_test", required=True, help="Path to test predictions parquet")
    ap.add_argument("--out", default="outputs/submission.csv", help="Output CSV path")

    ap.add_argument("--k", type=float, default=6)
    ap.add_argument("--vol_span", type=int, default=21)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--use_conf", action="store_true")

    args = ap.parse_args()

    cfg = SizingConfig(
        k=args.k,
        vol_span=args.vol_span,
        alpha=args.alpha,
        tau=args.tau,
        lam=args.lam,
        use_conf=args.use_conf,
    )

    submit_allocations(
        train_csv=args.train,
        test_csv=args.test,
        preds_test_parquet=args.preds_test,
        out_path=args.out,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()
