import os
from typing import Dict, Tuple, List
import pandas as pd

from arb_backtest import StrategyConfig, run_backtest, find_all_inputs, save_outputs


def run_cfg(root: str, label: str, params: Dict, out_root: str) -> Tuple[str, Dict[str, float]]:
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
    out_dir = os.path.join(out_root, label)
    save_outputs(df, stats, out_dir)
    return out_dir, stats


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # A/B: с дивидендами vs без дивидендов
    ab_root = os.path.join(root, "output", "dividend_AB")
    os.makedirs(ab_root, exist_ok=True)
    with_params = dict(dividend_yield_annual=0.12)
    without_params = dict(dividend_yield_annual=0.0)
    out_with, stats_with = run_cfg(root, "with_dividends", with_params, ab_root)
    out_without, stats_without = run_cfg(root, "without_dividends", without_params, ab_root)

    impact = pd.DataFrame([
        {"variant": "with", **stats_with},
        {"variant": "without", **stats_without},
        {"variant": "delta", **{k: (stats_with[k] - stats_without[k]) for k in stats_with.keys()}},
    ])
    impact.to_csv(os.path.join(ab_root, "dividend_impact.csv"), index=False)

    # Прогон по сетке дивидендных ставок
    sweep_root = os.path.join(root, "output", "dividend_sweep")
    os.makedirs(sweep_root, exist_ok=True)
    yields: List[float] = [0.0, 0.06, 0.09, 0.12, 0.15]
    rows = []
    for y in yields:
        _, st = run_cfg(root, f"yield_{int(y*100)}pct", dict(dividend_yield_annual=y), sweep_root)
        rows.append({"dividend_yield_annual": y, **st})
    pd.DataFrame(rows).to_csv(os.path.join(sweep_root, "dividend_sweep.csv"), index=False)

    print("A/B ->", ab_root)
    print("Sweep ->", sweep_root)


if __name__ == "__main__":
    main()
