import os
from typing import Tuple

import pandas as pd

from arb_backtest import (
    StrategyConfig,
    run_backtest,
    find_all_inputs,
    load_moex_minute_file,
    save_outputs,
)


CRISIS_DATE = pd.Timestamp("2025-04-01").date()


def apply_linear_scale(series: pd.Series, start: float, end: float) -> pd.Series:
    n = len(series)
    if n <= 1:
        return series * end
    factors = pd.Series([start + (end - start) * i / (n - 1) for i in range(n)], index=series.index)
    return series * factors


def apply_crisis_scenario(sber_df: pd.DataFrame, ofz_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = sber_df.copy()
    b = ofz_df.copy()

    # Выбрать строки кризисного дня
    s_day = s[pd.to_datetime(s["dt"]).dt.date == CRISIS_DATE].copy()
    b_day = b[pd.to_datetime(b["dt"]).dt.date == CRISIS_DATE].copy()

    if not s_day.empty:
        # Двухэтапное падение по Сберу: первые 60 минут до −10%, затем до −20%
        n = len(s_day)
        k = min(60, n)
        first = s_day.iloc[:k]
        rest = s_day.iloc[k:]
        if not first.empty:
            for col in ["open", "high", "low", "close"]:
                s.loc[first.index, col] = apply_linear_scale(first[col], 1.0, 0.9)
        if not rest.empty:
            for col in ["open", "high", "low", "close"]:
                s.loc[rest.index, col] = apply_linear_scale(rest[col], 0.9, 0.8)

    if not b_day.empty:
        # Мягкое движение по ОФЗ: линейно до −0.5% за день
        for col in ["open", "high", "low", "close"]:
            b.loc[b_day.index, col] = apply_linear_scale(b_day[col], 1.0, 0.995)

    return s, b


def to_moex_semicolon(df: pd.DataFrame, out_path: str, ticker: str) -> None:
    tmp = df.copy()
    tmp["DATE"] = pd.to_datetime(tmp["dt"]).dt.strftime("%Y%m%d")
    tmp["TIME"] = pd.to_datetime(tmp["dt"]).dt.strftime("%H%M%S")
    tmp["TICKER"] = ticker
    tmp["PER"] = 1
    tmp["VOL"] = tmp.get("volume", 0)
    tmp["OPENINT"] = tmp.get("open_interest", 0)
    cols = ["TICKER", "PER", "DATE", "TIME", "open", "high", "low", "close", "VOL", "OPENINT"]
    tmp = tmp[cols]
    tmp.to_csv(out_path, sep=";", index=False, header=True)


def run_case(label: str, cfg: StrategyConfig, sber_path: str, ofz_path: str, out_root: str) -> Tuple[str, dict]:
    df, stats = run_backtest(sber_path, ofz_path, cfg)
    out_dir = os.path.join(out_root, label)
    save_outputs(df, stats, out_dir)
    return out_dir, stats


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    inputs = find_all_inputs(root)
    sber_src = inputs.get("SBER")
    ofz_src = inputs.get("OFZ_26240") or inputs.get("OFZ_26241")
    if not sber_src or not ofz_src:
        raise FileNotFoundError("Source SBER/OFZ files not found.")

    # Загрузить исходные данные
    sber_df = load_moex_minute_file(sber_src)
    ofz_df = load_moex_minute_file(ofz_src)

    # Применить кризисный шок к данным
    sber_cr, ofz_cr = apply_crisis_scenario(sber_df, ofz_df)

    # Сохранить модифицированные входы в output/crysis_test/data
    out_root = os.path.join(root, "output", "crysis_test")
    data_dir = os.path.join(out_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    sber_path = os.path.join(data_dir, "SBER_crysis.txt")
    ofz_path = os.path.join(data_dir, "OFZ_26240_crysis.txt")
    to_moex_semicolon(sber_cr, sber_path, ticker="SBER")
    to_moex_semicolon(ofz_cr, ofz_path, ticker="OFZ")

    # Базовая конфигурация
    base_cfg = StrategyConfig(z_lookback_min=120, z_enter=1.5, z_exit=0.25, hedge_mode="OFZ_SINGLE")
    out_dir, stats = run_case("baseline", base_cfg, sber_path, ofz_path, out_root)
    print("crysis baseline ->", out_dir, stats)

    # С шок‑халтом 2%
    shock_cfg = StrategyConfig(z_lookback_min=120, z_enter=1.5, z_exit=0.25, hedge_mode="OFZ_SINGLE",
                               enable_shock_halt=True, shock_window_min=5, shock_move_threshold=0.02, shock_cooldown_min=10)
    out_dir2, stats2 = run_case("shock_halt_2pct", shock_cfg, sber_path, ofz_path, out_root)
    print("crysis shock_halt_2pct ->", out_dir2, stats2)

    # С дневным PnL‑стопом 0.5 млн
    stop_cfg = StrategyConfig(z_lookback_min=120, z_enter=1.5, z_exit=0.25, hedge_mode="OFZ_SINGLE",
                              enable_daily_pnl_stop=True, daily_pnl_stop_rub=500_000.0)
    out_dir3, stats3 = run_case("daily_stop_0_5m", stop_cfg, sber_path, ofz_path, out_root)
    print("crysis daily_stop_0_5m ->", out_dir3, stats3)


if __name__ == "__main__":
    main()


