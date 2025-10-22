import os

from arb_backtest import StrategyConfig, run_backtest, find_all_inputs, save_outputs


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    inputs = find_all_inputs(root)
    sber_path = inputs.get("SBER")
    ofz40_path = inputs.get("OFZ_26240")
    ofz41_path = inputs.get("OFZ_26241")
    rgbi_path = inputs.get("RGBI")

    # Использовать best-PnL (1‑мин) как базу и включить волатильностные выключатели
    cfg = StrategyConfig(
        z_lookback_min=120,
        z_enter=1.5,
        z_exit=0.25,
        timeframe=None,
        hedge_mode="OFZ_SINGLE",
        enable_vol_halt=True,
        vol_halt_mode="no_new",
        vol_halt_window_min=240,
        vol_halt_threshold=0.02,
    )

    df, stats = run_backtest(sber_path, ofz40_path, cfg, ofz41_path=ofz41_path, rgbi_path=rgbi_path)
    out_dir = os.path.join(root, "output", "best_pnl_1min_ofz_single_halt")
    save_outputs(df, stats, out_dir)

    print("Vol-halt run completed ->", out_dir)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


