import os
from typing import Dict, Tuple

from arb_backtest import StrategyConfig, run_backtest, find_all_inputs, save_outputs


CASES = [
    ("pnl_stop_0_3m", dict(enable_daily_pnl_stop=True, daily_pnl_stop_rub=300_000.0)),
    ("pnl_stop_0_5m", dict(enable_daily_pnl_stop=True, daily_pnl_stop_rub=500_000.0)),
    ("pnl_stop_0_7m", dict(enable_daily_pnl_stop=True, daily_pnl_stop_rub=700_000.0)),
    ("pnl_stop_1m",   dict(enable_daily_pnl_stop=True, daily_pnl_stop_rub=1_000_000.0)),
    ("pnl_trailing_0_7m", dict(enable_trailing_pnl_stop=True, trailing_pnl_stop_rub=700_000.0)),
    ("pnl_trailing_1m",   dict(enable_trailing_pnl_stop=True, trailing_pnl_stop_rub=1_000_000.0)),
]


def run_case(root: str, label: str, params: Dict) -> Tuple[str, Dict[str, float]]:
    inputs = find_all_inputs(root)
    sber_path = inputs.get("SBER")
    ofz40_path = inputs.get("OFZ_26240")
    ofz41_path = inputs.get("OFZ_26241")
    rgbi_path = inputs.get("RGBI")

    cfg = StrategyConfig(
        z_lookback_min=120,
        z_enter=1.5,
        z_exit=0.25,
        timeframe=None,
        hedge_mode="OFZ_SINGLE",
        **params,
    )

    df, stats = run_backtest(sber_path, ofz40_path, cfg, ofz41_path=ofz41_path, rgbi_path=rgbi_path)
    out_dir = os.path.join(root, "output", "pnl_stop", label)
    save_outputs(df, stats, out_dir)
    return out_dir, stats


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    for label, params in CASES:
        out, stats = run_case(root, label, params)
        print(f"{label} -> {out} :: {stats}")


if __name__ == "__main__":
    main()
