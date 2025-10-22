import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class StrategyConfig:
    # Торговые ограничения
    gross_limit_rub: float = 100_000_000.0  # 100 млн ₽
    # Издержки и финансирование
    # По ТЗ: 1 б.п. = 0.0001%, т.е. 1e-6 от оборота на каждую ногу сделки
    commission_fraction: float = 1e-6
    key_rate_annual: float = 0.15  # заём в ₽ по ключевой ставке (возьмем 15%)
    deposit_spread: float = 0.01  # размещение по ставке: КС − 1%
    dividend_yield_annual: float = 0.12  # ожидаемая годовая дивидендная доходность по Сберу
    # Сигналы
    beta_lookback_min: int = 120  # окно расчёта β (минуты)
    z_lookback_min: int = 240  # окно расчёта статистики z-score (минуты)
    z_enter: float = 2.0
    z_exit: float = 0.5
    # Адаптация к волатильности и рыночным режимам
    vol_lookback_min: int = 240
    vol_target: float = 0.005  # целевая минутная волатильность спреда (0.5%) для масштабирования риска
    max_leverage_scale: float = 1.0  # верхняя граница множителя масштабирования
    min_leverage_scale: float = 0.25  # нижняя граница множителя масштабирования
    # Режим масштабирования по волатильности: 'inverse' (по умолчанию) или 'direct'; степень alpha
    vol_scale_mode: str = "inverse"
    vol_scale_alpha: float = 1.0
    # Применять direct‑масштабирование только в верхней корзине волы (топ‑квантиль)
    vol_scale_direct_top_bucket_only: bool = False
    vol_top_quantile: float = 0.75
    # Режимы рынка
    regime_lookback_min: int = 60
    regime_flat_thr: float = 0.001
    enable_regime_adapt: bool = False
    # Мультипликаторы по режимам: размер позиции и пороги вход/выход
    regime_size_multipliers: Dict[str, float] = None  # заполняется по умолчанию в runtime
    regime_z_enter_multipliers: Dict[str, float] = None
    regime_z_exit_multipliers: Dict[str, float] = None
    # Выключатели торговли по волатильности спреда
    enable_vol_halt: bool = False  # если True — активировать выключатели
    vol_halt_mode: str = "no_new"  # 'no_new' — не открывать новые позиции/перевороты; 'flat' — принудительно в ноль
    vol_halt_window_min: int = 240  # окно оценки реализованной волатильности спреда
    vol_halt_threshold: float = 0.02  # порог минутной волатильности спреда (например, 2%)
    vol_halt_quantile: Optional[float] = None  # если задано (напр., 0.95), порог = квантиль spread_vol
    vol_halt_condition: str = "above"  # 'above' — глушить при высокой воле; 'below' — при низкой
    # Халт на шоковые движения
    enable_shock_halt: bool = False
    shock_window_min: int = 5
    shock_move_threshold: float = 0.02  # 2% движение по Сберу за окно
    shock_cooldown_min: int = 0  # минуты удержания вне позиции после шока
    # Стопы по PnL
    enable_daily_pnl_stop: bool = False  # дневной лимит убытка (flat до конца дня)
    daily_pnl_stop_rub: float = 0.0  # например, 500_000.0
    daily_pnl_stop_cooldown_min: int = 0  # опционально удерживать flat N минут следующего дня старта (0 = до конца дня)
    enable_trailing_pnl_stop: bool = False  # стоп по трейлинговой просадке
    trailing_pnl_stop_rub: float = 0.0  # например, 1_000_000.0
    trailing_stop_cooldown_min: int = 0
    # Торговые часы (Москва). Если None, используем все метки времени из файла
    session_start: Optional[str] = "10:00:00"
    session_end: Optional[str] = "18:50:00"
    timezone: Optional[str] = None  # по умолчанию оставляем наивные метки времени
    # Таймфрейм агрегации: None или '5T' (5 минут)
    timeframe: Optional[str] = None
    # Режим хеджа: 'OFZ_SINGLE' | 'OFZ_BASKET' | 'RGBI'
    hedge_mode: str = "OFZ_SINGLE"
    # Веса корзины для ОФЗ 26240/26241
    ofz_weights: Tuple[float, float] = (0.5, 0.5)
    # Предпочитаемая одиночная ОФЗ: '26240' или '26241'
    prefer_ofz: str = "26240"


def _infer_file_role(filename: str) -> Optional[str]:
    name = os.path.basename(filename)
    if "Сбербанк" in name or "Sber" in name or "SBER" in name:
        return "SBER"
    if "26240" in name:
        return "OFZ_26240"
    if "26241" in name:
        return "OFZ_26241"
    if "RGBI" in name.upper() or "RGBI" in name:
        return "RGBI"
    return None


def load_moex_minute_file(path: str) -> pd.DataFrame:
    """Загрузка минутных баров с разделителем ';' и русской шапкой.

    Ожидаемые столбцы: <TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>;<OPENINT>
    Форматы: DATE — YYYYMMDD, TIME — HHMMSS.
    """
    df = pd.read_csv(path, sep=";", header=0)
    # Приведение названий столбцов к стандартным
    df.columns = [
        "ticker",
        "per",
        "date",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]

    # Парсинг даты/времени
    date_str = df["date"].astype(str).str.zfill(8)
    time_str = df["time"].astype(str).str.zfill(6)
    dt = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M%S", errors="coerce")
    if StrategyConfig.timezone:
        # По умолчанию оставляем наивные метки, если часовой пояс не указан
        dt = dt.dt.tz_localize(StrategyConfig.timezone, nonexistent="shift_forward", ambiguous="NaT")
    df["dt"] = dt
    df = df.drop(columns=["date", "time"]).sort_values("dt").reset_index(drop=True)

    # Фильтрация 1‑минутных баров
    df = df[df["per"] == 1].copy()

    # Удаление дубликатов, оставить последний
    df = df.drop_duplicates(subset=["dt"], keep="last")

    # Оставить только необходимые столбцы
    return df[["dt", "ticker", "open", "high", "low", "close", "volume", "open_interest"]]


def clip_to_session(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is None or end is None:
        return df
    intraday = df.copy()
    intraday["time"] = intraday["dt"].dt.time.astype(str)
    intraday = intraday[(intraday["time"] >= start) & (intraday["time"] <= end)].copy()
    intraday = intraday.drop(columns=["time"]).reset_index(drop=True)
    return intraday


def aggregate_timeframe(df: pd.DataFrame, timeframe: Optional[str]) -> pd.DataFrame:
    """Агрегация синхронизированных 1‑минутных данных в более крупный таймфрейм (например, '5T').

    Агрегирует OHLC и объёмы по каждой ноге. В метке 'dt' сохраняется начало корзины.
    """
    if not timeframe or timeframe in ("1T", "1min"):
        return df
    df2 = df.copy()
    df2["bucket"] = df2["dt"].dt.floor(timeframe)
    # Агрегировать только те столбцы, которые присутствуют
    agg: Dict[str, str] = {}
    for col in [
        "sber_open", "sber_high", "sber_low", "sber_close", "sber_volume", "sber_open_interest",
        "ofz_open", "ofz_high", "ofz_low", "ofz_close", "ofz_volume", "ofz_open_interest",
        "hedge_close", "ofz40_close", "ofz41_close",
    ]:
        if col in df2.columns:
            if col.endswith("_open"):
                agg[col] = "first"
            elif col.endswith("_high"):
                agg[col] = "max"
            elif col.endswith("_low"):
                agg[col] = "min"
            elif col.endswith("_volume"):
                agg[col] = "sum"
            else:
                agg[col] = "last"
    g = df2.groupby("bucket", as_index=False).agg(agg)
    g = g.rename(columns={"bucket": "dt"})
    # Определить обязательные столбцы для dropna
    required = ["sber_close"]
    if "hedge_close" in g.columns:
        required.append("hedge_close")
    elif "ofz40_close" in g.columns and "ofz41_close" in g.columns:
        required.extend(["ofz40_close", "ofz41_close"])
    return g.dropna(subset=required).reset_index(drop=True)


def compute_rolling_beta(df: pd.DataFrame, lookback: int) -> pd.Series:
    s_r = df["sber_close"].pct_change()
    b_r = df["ofz_close"].pct_change()
    cov = s_r.rolling(lookback, min_periods=max(10, lookback // 5)).cov(b_r)
    var = b_r.rolling(lookback, min_periods=max(10, lookback // 5)).var()
    beta = cov / var.replace(0.0, np.nan)
    return beta.ffill().fillna(0.0)


def compute_zscore_spread(df: pd.DataFrame, beta: pd.Series, lookback: int) -> pd.Series:
    s_r = df["sber_close"].pct_change()
    b_r = df["ofz_close"].pct_change()
    spread = s_r - beta * b_r
    mean = spread.rolling(lookback, min_periods=max(20, lookback // 5)).mean()
    std = spread.rolling(lookback, min_periods=max(20, lookback // 5)).std(ddof=0)
    z = (spread - mean) / (std.replace(0.0, np.nan))
    return z.fillna(0.0)


def compute_vol_scale(series: pd.Series, target: float, lookback: int, floor: float, cap: float) -> pd.Series:
    vol = series.rolling(lookback, min_periods=max(20, lookback // 5)).std(ddof=0)
    scale = (target / vol.replace(0.0, np.nan)).clip(lower=floor, upper=cap)
    return scale.fillna(floor)

def compute_vol_scale_general(series: pd.Series, target: float, lookback: int, floor: float, cap: float,
                              mode: str = "inverse", alpha: float = 1.0) -> pd.Series:
    """Обобщённое масштабирование по воле: inverse (target/vol)^alpha или direct (vol/target)^alpha, с клиппингом.

    По умолчанию эквивалентно прежней логике (inverse, alpha=1.0).
    """
    vol = series.rolling(lookback, min_periods=max(20, lookback // 5)).std(ddof=0)
    v = vol.replace(0.0, np.nan)
    mode_l = (mode or "inverse").lower()
    if mode_l == "direct":
        raw = (v / target)
    else:
        raw = (target / v)
    # степень alpha
    try:
        raw = raw ** alpha
    except Exception:
        pass
    scale = raw.clip(lower=floor, upper=cap)
    return scale.fillna(floor)


def detect_regime(prices: pd.Series, lookback: int = 60, flat_thr: float = 0.001) -> pd.Series:
    ret = prices.pct_change(lookback)
    regime = pd.Series(index=prices.index, dtype="object")
    regime[:] = "sideways"
    regime[ret > flat_thr] = "bull"
    regime[ret < -flat_thr] = "bear"
    return regime.fillna("sideways")


def run_backtest(
    sber_path: str,
    ofz_path: Optional[str],
    config: StrategyConfig,
    ofz41_path: Optional[str] = None,
    rgbi_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    sber = load_moex_minute_file(sber_path)
    sber = clip_to_session(sber, config.session_start, config.session_end)

    hedge_mode = (config.hedge_mode or "OFZ_SINGLE").upper()

    if hedge_mode == "RGBI":
        if not rgbi_path:
            raise FileNotFoundError("RGBI path not provided for RGBI hedge mode.")
        rgbi = load_moex_minute_file(rgbi_path)
        rgbi = clip_to_session(rgbi, config.session_start, config.session_end)
        merged = pd.merge(
            sber.rename(columns={"dt": "dt", "close": "sber_close"})[["dt", "sber_close"]],
            rgbi.rename(columns={"dt": "dt", "close": "hedge_close"})[["dt", "hedge_close"]],
            on="dt", how="inner"
        )
        data = merged
    elif hedge_mode == "OFZ_BASKET":
        if not ofz_path or not ofz41_path:
            raise FileNotFoundError("Both OFZ 26240 and 26241 paths are required for basket mode.")
        ofz40 = load_moex_minute_file(ofz_path)
        ofz41 = load_moex_minute_file(ofz41_path)
        ofz40 = clip_to_session(ofz40, config.session_start, config.session_end)
        ofz41 = clip_to_session(ofz41, config.session_start, config.session_end)
        m1 = pd.merge(
            sber.rename(columns={"dt": "dt", "close": "sber_close"})[["dt", "sber_close"]],
            ofz40.rename(columns={"dt": "dt", "close": "ofz40_close"})[["dt", "ofz40_close"]],
            on="dt", how="inner"
        )
        data = pd.merge(
            m1,
            ofz41.rename(columns={"dt": "dt", "close": "ofz41_close"})[["dt", "ofz41_close"]],
            on="dt", how="inner"
        )
        w40, w41 = config.ofz_weights
        total_w = (w40 + w41) if (w40 + w41) != 0 else 1.0
        w40 /= total_w
        w41 /= total_w
        data["hedge_close"] = w40 * data["ofz40_close"] + w41 * data["ofz41_close"]
    else:  # OFZ_SINGLE
        if not ofz_path and not ofz41_path:
            raise FileNotFoundError("OFZ path not provided for single hedge mode.")
        use_26240 = (config.prefer_ofz == "26240")
        chosen = ofz_path if use_26240 and ofz_path else (ofz41_path if ofz41_path else ofz_path)
        ofz = load_moex_minute_file(chosen)
        ofz = clip_to_session(ofz, config.session_start, config.session_end)
        merged = pd.merge(
            sber.rename(columns={"dt": "dt", "close": "sber_close"})[["dt", "sber_close"]],
            ofz.rename(columns={"dt": "dt", "close": "hedge_close"})[["dt", "hedge_close"]],
            on="dt", how="inner"
        )
        data = merged

    # Необязательная агрегация
    data = aggregate_timeframe(data, config.timeframe)
    if data.empty:
        raise ValueError("No overlapping minutes between Sber and OFZ after synchronization.")

    # Сигналы
    # Построить временный DataFrame в ожидаемом формате для β/z, используя hedge_close
    tmp = pd.DataFrame({
        "sber_close": data["sber_close"].values,
        "ofz_close": data["hedge_close"].values,
    }, index=data.index)
    beta = compute_rolling_beta(tmp, config.beta_lookback_min)
    z = compute_zscore_spread(tmp, beta, config.z_lookback_min)
    # Реализованная волатильность спреда r_S − β·r_H для выключателей и анализа
    s_r = tmp["sber_close"].pct_change()
    h_r = tmp["ofz_close"].pct_change()
    spread = s_r - beta * h_r
    spread_vol = spread.rolling(config.vol_halt_window_min, min_periods=max(20, config.vol_halt_window_min // 5)).std(ddof=0)
    # Масштабирование нагрузки: по умолчанию 'inverse' (совместимо с прежними результатами)
    if config.vol_scale_direct_top_bucket_only:
        inv_scale = compute_vol_scale_general(
            z,
            config.vol_target,
            config.vol_lookback_min,
            config.min_leverage_scale,
            config.max_leverage_scale,
            mode="inverse",
            alpha=config.vol_scale_alpha,
        )
        dir_scale = compute_vol_scale_general(
            z,
            config.vol_target,
            config.vol_lookback_min,
            config.min_leverage_scale,
            config.max_leverage_scale,
            mode="direct",
            alpha=config.vol_scale_alpha,
        )
        # верхний квантиль по spread_vol
        try:
            top_thr = float(np.nanquantile(spread_vol.values.astype(float), config.vol_top_quantile))
        except Exception:
            top_thr = None
        if top_thr is None:
            vol_scale = inv_scale
        else:
            mask = (spread_vol >= top_thr)
            vol_scale = inv_scale.copy()
            vol_scale[mask] = dir_scale[mask]
    else:
        vol_scale = compute_vol_scale_general(
            z,
            config.vol_target,
            config.vol_lookback_min,
            config.min_leverage_scale,
            config.max_leverage_scale,
            mode=config.vol_scale_mode,
            alpha=config.vol_scale_alpha,
        )

    # Определение рыночного режима (на базе Сбера для простоты)
    regime = detect_regime(data["sber_close"], lookback=config.regime_lookback_min, flat_thr=config.regime_flat_thr)

    # Инициализация
    df = data.copy()
    df["beta"] = beta.values
    df["z"] = z.values
    df["vol_scale"] = vol_scale.values
    df["spread_vol"] = spread_vol.values
    df["regime"] = regime.values

    # Подготовить адаптивные пороги по режимам (если включено)
    z_enter_arr = np.full(len(df), config.z_enter, dtype=float)
    z_exit_arr = np.full(len(df), config.z_exit, dtype=float)
    if config.enable_regime_adapt:
        # Значения по умолчанию
        size_mult = config.regime_size_multipliers or {"bull": 0.9, "bear": 1.1, "sideways": 1.0}
        z_ent_mult = config.regime_z_enter_multipliers or {"bull": 1.1, "bear": 1.1, "sideways": 1.0}
        z_ex_mult = config.regime_z_exit_multipliers or {"bull": 1.0, "bear": 1.0, "sideways": 1.0}
        # Применить мультипликаторы к z‑порогам и vol_scale позже к размеру
        reg = df["regime"].astype(str).values
        for i, r in enumerate(reg):
            z_enter_arr[i] = config.z_enter * z_ent_mult.get(r, 1.0)
            z_exit_arr[i] = config.z_exit * z_ex_mult.get(r, 1.0)
        # Обновим vol_scale под размерные мультипликаторы по режиму
        df["vol_scale"] = df["vol_scale"] * df["regime"].map(lambda r: size_mult.get(str(r), 1.0)).astype(float)

    # Направление позиции: +1 — long‑спред (long Сбер, short ОФЗ), −1 — short‑спред
    pos_dir = np.where(df["z"] < -z_enter_arr, 1,
                np.where(df["z"] > z_enter_arr, -1, np.nan))
    # Держим до сигнала выхода
    pos_dir_series = pd.Series(pos_dir, index=df.index)
    pos_dir_series = pos_dir_series.ffill()
    # Выход при |z| < z_exit (адаптивный порог)
    exit_mask = df["z"].abs() < z_exit_arr
    pos_dir_series[exit_mask] = 0
    pos_dir_series = pos_dir_series.ffill().fillna(0)
    df["pos_dir"] = pos_dir_series

    # Выключатели торговли по волатильности спреда
    # Рассчитать эффективный порог для выключателя: квантильный или фиксированный
    eff_threshold = None
    if config.enable_vol_halt:
        if config.vol_halt_quantile is not None:
            try:
                eff_threshold = float(np.nanquantile(df["spread_vol"].values.astype(float), config.vol_halt_quantile))
            except Exception:
                eff_threshold = None
        if eff_threshold is None and (config.vol_halt_threshold or 0) > 0:
            eff_threshold = float(config.vol_halt_threshold)

    if config.enable_vol_halt and eff_threshold is not None:
        if str(config.vol_halt_condition).lower() == "below":
            halted = (df["spread_vol"] < eff_threshold)
        else:
            halted = (df["spread_vol"] > eff_threshold)
        pos_prev = df["pos_dir"].shift(1)
        pos_new = df["pos_dir"].copy()
        if config.vol_halt_mode.lower() == "flat":
            # Принудительно закрыть позиции в периоды экстремальной волатильности
            pos_new[halted] = 0
        else:
            # Блокировать только входы и перевороты; выходы в ноль разрешены
            is_entry = (pos_prev.fillna(0) == 0) & (pos_new != 0)
            is_flip = (pos_prev.fillna(0) != 0) & (pos_new != 0) & (np.sign(pos_prev.fillna(0)) != np.sign(pos_new))
            freeze = halted & (is_entry | is_flip)
            pos_new[freeze] = pos_prev[freeze]
            # Разрешить выходы (pos_new==0), даже если halted
            # Нет дополнительной обработки — они остаются как есть
        df["pos_dir"] = pos_new.ffill().fillna(0)

    # Халт на шоковые движения по Сберу
    if config.enable_shock_halt and (config.shock_move_threshold or 0) > 0:
        s_move = df["sber_close"].pct_change(config.shock_window_min).abs().fillna(0.0)
        shock = s_move > config.shock_move_threshold
        if config.shock_cooldown_min and config.shock_cooldown_min > 0:
            # Распространить шок на последующие минуты (cooldown)
            shock_idx = np.where(shock.values)[0]
            mark = np.zeros(len(df), dtype=bool)
            for idx in shock_idx:
                end = min(len(df), idx + 1 + config.shock_cooldown_min)
                mark[idx:end] = True
            shock = pd.Series(mark, index=df.index)
        pos_shock = df["pos_dir"].copy()
        pos_shock[shock] = 0
        df["pos_dir"] = pos_shock

    # Дискретные изменения состояния позиции
    df["pos_change"] = df["pos_dir"].diff().fillna(df["pos_dir"])  # ненулевое -> вход/выход/переворот

    # Предварительный расчёт для PnL-стопов (на базе текущих позиций)
    def _compute_pnl_components(pos_dir_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        # Динамическое масштабирование валовой нагрузки по vol_scale
        gross = 0.5 * config.gross_limit_rub * df["vol_scale"]
        s_qty = (gross / df["sber_close"]) * pos_dir_series
        if (config.hedge_mode or "OFZ_SINGLE").upper() == "OFZ_BASKET":
            w40, w41 = config.ofz_weights
            tw = (w40 + w41) if (w40 + w41) != 0 else 1.0
            w40 /= tw; w41 /= tw
            ofz40_qty = (gross * w40 / df["ofz40_close"]) * (-pos_dir_series)
            ofz41_qty = (gross * w41 / df["ofz41_close"]) * (-pos_dir_series)
        else:
            hedge_qty = (gross / df["hedge_close"]) * (-pos_dir_series)

        commission_rate = config.commission_fraction
        d_sqty = s_qty.diff().fillna(s_qty.abs())
        turn_s = d_sqty.abs() * df["sber_close"]
        if (config.hedge_mode or "OFZ_SINGLE").upper() == "OFZ_BASKET":
            d_q40 = ofz40_qty.diff().fillna(ofz40_qty.abs())
            d_q41 = ofz41_qty.diff().fillna(ofz41_qty.abs())
            turn40 = d_q40.abs() * df["ofz40_close"]
            turn41 = d_q41.abs() * df["ofz41_close"]
            comm = commission_rate * (turn_s + turn40 + turn41)
        else:
            d_hq = hedge_qty.diff().fillna(hedge_qty.abs())
            turn_h = d_hq.abs() * df["hedge_close"]
            comm = commission_rate * (turn_s + turn_h)

        d_s = df["sber_close"].diff().fillna(0.0)
        if (config.hedge_mode or "OFZ_SINGLE").upper() == "OFZ_BASKET":
            d40 = df["ofz40_close"].diff().fillna(0.0)
            d41 = df["ofz41_close"].diff().fillna(0.0)
            price_pnl = s_qty.shift(1).fillna(0.0) * d_s + ofz40_qty.shift(1).fillna(0.0) * d40 + ofz41_qty.shift(1).fillna(0.0) * d41
        else:
            d_h = df["hedge_close"].diff().fillna(0.0)
            price_pnl = s_qty.shift(1).fillna(0.0) * d_s + hedge_qty.shift(1).fillna(0.0) * d_h

        minutes_per_year = 365 * 24 * 60
        br = config.key_rate_annual / minutes_per_year
        dr = max(config.key_rate_annual - config.deposit_spread, 0.0) / minutes_per_year
        long_cash = gross
        short_cash = gross
        carry = (-br * long_cash + dr * short_cash) * (pos_dir_series.abs() > 0).astype(float)

        div_rate_min = config.dividend_yield_annual / minutes_per_year
        s_long_not = (s_qty.shift(1).clip(lower=0.0) * df["sber_close"]).fillna(0.0)
        s_short_not = (s_qty.shift(1).clip(upper=0.0).abs() * df["sber_close"]).fillna(0.0)
        div_c = div_rate_min * (s_long_not - s_short_not)

        total = price_pnl + carry + (-comm)
        return total, s_qty, (ofz40_qty if (config.hedge_mode or "OFZ_SINGLE").upper()=="OFZ_BASKET" else hedge_qty), comm

    # Если включены PnL-стопы — предварительно оценим триггеры и скорректируем pos_dir
    if (config.enable_daily_pnl_stop and config.daily_pnl_stop_rub > 0.0) or (config.enable_trailing_pnl_stop and config.trailing_pnl_stop_rub > 0.0):
        prelim_total, _, _, _ = _compute_pnl_components(df["pos_dir"])  # предварительный PnL при текущих позициях
        prelim_cum = prelim_total.cumsum()
        # Дневной стоп: после превышения убытка по дню — flat до конца дня
        halted_mask = pd.Series(False, index=df.index)
        if config.enable_daily_pnl_stop and config.daily_pnl_stop_rub > 0.0:
            dates = pd.to_datetime(df["dt"]).dt.date
            day_ids = pd.Series(dates, index=df.index)
            # дневной PnL как cumsum по дню
            day_group = day_ids
            day_cum = prelim_total.groupby(day_group).cumsum()
            # найдём первое пересечение порога в каждом дне и заморозим остаток дня
            for day, sub in day_cum.groupby(day_group):
                idx = sub.index
                crossed = sub < (-config.daily_pnl_stop_rub)
                if crossed.any():
                    first = crossed.idxmax() if crossed.any() else None
                    if first is not None and crossed.loc[first]:
                        halted_mask.loc[first:] = halted_mask.loc[first:] | (day_ids.loc[first:] == day)
        # Трейлинговый стоп: если просадка от исторического максимума превысила порог — flat (с cooldown)
        if config.enable_trailing_pnl_stop and config.trailing_pnl_stop_rub > 0.0:
            dd = prelim_cum - prelim_cum.cummax()
            tr_cross = dd < (-config.trailing_pnl_stop_rub)
            if tr_cross.any():
                first_tr = tr_cross.idxmax() if tr_cross.any() else None
                if first_tr is not None and tr_cross.loc[first_tr]:
                    halted_mask.loc[first_tr:] = True
                    if config.trailing_stop_cooldown_min and config.trailing_stop_cooldown_min > 0:
                        # по умолчанию уже flat до конца, cooldown не требуется дополнительно
                        pass
        # Применить маску
        if halted_mask.any():
            pos_after_stops = df["pos_dir"].copy()
            pos_after_stops[halted_mask] = 0
            df["pos_dir"] = pos_after_stops

    # Динамическое масштабирование валовой нагрузки по vol_scale (по режимам можно доработать при желании)
    gross_per_side = 0.5 * config.gross_limit_rub * df["vol_scale"]

    # Рублёвая нейтральность: равные и противоположные рублёвые ноционалы
    # Long‑спред (+1): long Сбер, short ОФЗ. Short‑спред (−1): short Сбер, long ОФЗ.
    # Расчёт количества единиц (непрерывно) по каждой ноге
    sber_qty = (gross_per_side / df["sber_close"]) * df["pos_dir"]
    df["sber_qty"] = sber_qty
    hedge_mode = (config.hedge_mode or "OFZ_SINGLE").upper()
    if hedge_mode == "OFZ_BASKET":
        w40, w41 = config.ofz_weights
        total_w = (w40 + w41) if (w40 + w41) != 0 else 1.0
        w40 /= total_w
        w41 /= total_w
        df["ofz40_qty"] = (gross_per_side * w40 / df["ofz40_close"]) * (-df["pos_dir"])  # противоположная нога
        df["ofz41_qty"] = (gross_per_side * w41 / df["ofz41_close"]) * (-df["pos_dir"])  # противоположная нога
    else:
        df["hedge_qty"] = (gross_per_side / df["hedge_close"]) * (-df["pos_dir"])  # противоположная нога

    # Транзакционные издержки: на изменение абсолютного ноциона по каждой ноге
    commission_rate = config.commission_fraction
    d_qty_sber = df["sber_qty"].diff().fillna(df["sber_qty"].abs())
    turnover_sber = d_qty_sber.abs() * df["sber_close"]
    if hedge_mode == "OFZ_BASKET":
        d_qty_ofz40 = df["ofz40_qty"].diff().fillna(df["ofz40_qty"].abs())
        d_qty_ofz41 = df["ofz41_qty"].diff().fillna(df["ofz41_qty"].abs())
        turnover_ofz40 = d_qty_ofz40.abs() * df["ofz40_close"]
        turnover_ofz41 = d_qty_ofz41.abs() * df["ofz41_close"]
        commission = commission_rate * (turnover_sber + turnover_ofz40 + turnover_ofz41)
    else:
        d_qty_hedge = df["hedge_qty"].diff().fillna(df["hedge_qty"].abs())
        turnover_hedge = d_qty_hedge.abs() * df["hedge_close"]
        commission = commission_rate * (turnover_sber + turnover_hedge)

    # PnL от изменения цен
    d_sber = df["sber_close"].diff().fillna(0.0)
    if hedge_mode == "OFZ_BASKET":
        d_ofz40 = df["ofz40_close"].diff().fillna(0.0)
        d_ofz41 = df["ofz41_close"].diff().fillna(0.0)
        pnl_price = (
            df["sber_qty"].shift(1).fillna(0.0) * d_sber
            + df["ofz40_qty"].shift(1).fillna(0.0) * d_ofz40
            + df["ofz41_qty"].shift(1).fillna(0.0) * d_ofz41
        )
    else:
        d_hedge = df["hedge_close"].diff().fillna(0.0)
        pnl_price = df["sber_qty"].shift(1).fillna(0.0) * d_sber + df["hedge_qty"].shift(1).fillna(0.0) * d_hedge

    # Финансовый carry: заём по длинной ноге под ключевую ставку, размещение по короткой под (КС − 1%)
    # При рублёвой нейтральности (gross_per_side на каждой ноге) начисление в минуту:
    minutes_per_year = 365 * 24 * 60
    borrow_rate_min = config.key_rate_annual / minutes_per_year
    deposit_rate_min = max(config.key_rate_annual - config.deposit_spread, 0.0) / minutes_per_year
    long_cash = gross_per_side  # денежная нагрузка длинной ноги в минуту
    short_cash = gross_per_side  # поступления по короткой ноге
    carry = -borrow_rate_min * long_cash + deposit_rate_min * short_cash
    # Применяем только при наличии позиции
    carry = carry * (df["pos_dir"].abs() > 0).astype(float)

    # Дивидендный carry по Сберу: начислять при наличии позиции; знак зависит от long/short
    div_rate_min = config.dividend_yield_annual / minutes_per_year
    sber_long_notional = (df["sber_qty"].shift(1).clip(lower=0.0) * df["sber_close"]).fillna(0.0)
    sber_short_notional = (df["sber_qty"].shift(1).clip(upper=0.0).abs() * df["sber_close"]).fillna(0.0)
    div_carry = div_rate_min * (sber_long_notional - sber_short_notional)

    # Чистый PnL за минуту
    df["pnl_price"] = pnl_price
    df["pnl_carry"] = carry + div_carry
    df["pnl_commission"] = -commission
    df["pnl_total"] = df["pnl_price"] + df["pnl_carry"] + df["pnl_commission"]
    df["pnl_cum"] = df["pnl_total"].cumsum()
    df["pnl_cum_max"] = df["pnl_cum"].cummax()
    df["drawdown"] = (df["pnl_cum"] - df["pnl_cum"].cummax())

    # Показатели риска
    # Нормализуем доходность на лимит валовой позиции для получения псевдо‑ретёрна
    ret_series = df["pnl_total"] / config.gross_limit_rub
    ann_factor = np.sqrt(252 * 6.5 * 60)  # приблизительное число торговых минут в году для годовой шкалы
    sharpe = (ret_series.mean() / (ret_series.std(ddof=0) + 1e-12)) * ann_factor
    max_dd = (df["pnl_cum"] - df["pnl_cum"].cummax()).min()
    stats = {
        "pnl_total": float(df["pnl_total"].sum()),
        "pnl_carry": float(df["pnl_carry"].sum()),
        "pnl_commission": float(df["pnl_commission"].sum()),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
    }

    # Сохранить нормализованную доходность для последующих сводных отчётов
    df["ret"] = ret_series

    return df, stats


def find_input_files(root_dir: str) -> Tuple[str, str]:
    files = os.listdir(root_dir)
    sber_path = None
    ofz_path = None
    for f in files:
        role = _infer_file_role(f)
        if role == "SBER":
            sber_path = os.path.join(root_dir, f)
        elif role in ("OFZ_26240", "OFZ_26241"):
            # prefer 26240 by default
            if ofz_path is None or ("26240" in f and "26241" in os.path.basename(ofz_path)):
                ofz_path = os.path.join(root_dir, f)
    if sber_path is None or ofz_path is None:
        raise FileNotFoundError("Could not locate Sberbank and OFZ files in the project root.")
    return sber_path, ofz_path


def find_all_inputs(root_dir: str) -> Dict[str, Optional[str]]:
    files = os.listdir(root_dir)
    out: Dict[str, Optional[str]] = {"SBER": None, "OFZ_26240": None, "OFZ_26241": None, "RGBI": None}
    for f in files:
        role = _infer_file_role(f)
        if role in out and out[role] is None:
            out[role] = os.path.join(root_dir, f)
    return out


def _compute_top_drawdowns(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Расчёт топ‑эпизодов просадок с использованием сегментации по линии текущего максимума."""
    work = df[["dt", "pnl_cum", "pnl_cum_max"]].copy()
    # Идентификатор режима увеличивается при каждом новом ATH
    regime = (work["pnl_cum_max"] != work["pnl_cum_max"].shift(1)).cumsum()
    work["regime"] = regime
    episodes = []
    regime_ids = work["regime"].unique()
    for rid in regime_ids:
        seg = work[work["regime"] == rid]
        if seg.empty:
            continue
        peak_idx = seg.index.min()
        peak_time = work.loc[peak_idx, "dt"]
        peak_val = work.loc[peak_idx, "pnl_cum_max"]
        # минимум внутри сегмента
        trough_idx = seg["pnl_cum"].idxmin()
        trough_time = work.loc[trough_idx, "dt"]
        trough_val = work.loc[trough_idx, "pnl_cum"]
        depth = float(trough_val - peak_val)
        if depth >= 0:
            continue
        # восстановление: первое время позже, когда появится новый максимум выше этого пика
        later = work[work.index > seg.index.max()]
        rec = later[later["pnl_cum_max"] > peak_val]
        if not rec.empty:
            recovery_idx = rec.index.min()
            recovery_time = work.loc[recovery_idx, "dt"]
        else:
            recovery_idx = None
            recovery_time = None
        episodes.append({
            "peak_time": peak_time,
            "trough_time": trough_time,
            "recovery_time": recovery_time,
            "peak_val": float(peak_val),
            "trough_val": float(trough_val),
            "depth": depth,
        })
    epi_df = pd.DataFrame(episodes)
    if epi_df.empty:
        return epi_df
    epi_df = epi_df.sort_values("depth").head(top_n).reset_index(drop=True)
    return epi_df


def save_outputs(df: pd.DataFrame, stats: Dict[str, float], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "backtest_results.csv")
    df.to_csv(out_csv, index=False)
    # Экспорт компактной таблицы сигналов
    base_cols = [
        "dt", "z", "beta", "pos_dir", "pos_change", "sber_close", "pnl_price", "pnl_carry", "pnl_commission", "pnl_total", "pnl_cum"
    ]
    opt_cols = []
    if "hedge_close" in df.columns:
        opt_cols += ["hedge_close"]
    if "ofz40_close" in df.columns:
        opt_cols += ["ofz40_close"]
    if "ofz41_close" in df.columns:
        opt_cols += ["ofz41_close"]
    qty_cols = ["sber_qty"]
    for q in ["hedge_qty", "ofz40_qty", "ofz41_qty"]:
        if q in df.columns:
            qty_cols.append(q)
    signals_cols = base_cols[:5] + opt_cols + ["sber_close"] + qty_cols + base_cols[6:]
    # Обеспечить уникальность и сохранить порядок
    seen = set()
    ordered = []
    for c in signals_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            ordered.append(c)
    signals_path = os.path.join(out_dir, "signals.csv")
    df[ordered].to_csv(signals_path, index=False)
    # Также сохранить компактный файл со статистикой
    stats_path = os.path.join(out_dir, "stats.txt")
    with open(stats_path, "w", encoding="utf-8") as fh:
        for k, v in stats.items():
            fh.write(f"{k}: {v}\n")

    # Сохранить график PnL с метками входов/выходов
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["dt"], df["pnl_cum"], label="Cumulative PnL (RUB)", color="#1f77b4")
    # Линия текущего максимума
    ax.plot(df["dt"], df["pnl_cum_max"], color="#AAAAAA", linewidth=1.0, label="Running Max")
    # Заштриховать области просадок
    dd = df["drawdown"].values
    ax.fill_between(df["dt"], df["pnl_cum"], df["pnl_cum_max"], where=(dd < 0), color="#ff7f0e", alpha=0.15, interpolate=True, label="Drawdown")

    # Отметить входы/выходы
    entries = df[(df["pos_change"].abs() > 0) & (df["pos_dir"].abs() > 0)]
    exits = df[(df["pos_change"].abs() > 0) & (df["pos_dir"].abs() == 0)]
    ax.scatter(entries["dt"], entries["pnl_cum"], color="#2ca02c", s=10, label="Entry")
    ax.scatter(exits["dt"], exits["pnl_cum"], color="#d62728", s=10, label="Exit")

    ax.set_title("Arb Strategy Cumulative PnL with Entries/Exits")
    ax.set_xlabel("Time")
    ax.set_ylabel("PnL (RUB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pnl.png"))
    plt.close(fig)

    # Топ‑просадки: таблица и отметки
    top_dd = _compute_top_drawdowns(df, top_n=5)
    if not top_dd.empty:
        top_dd.to_csv(os.path.join(out_dir, "top_drawdowns.csv"), index=False)
        # Отметить впадины на отдельном графике для наглядности
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["dt"], df["pnl_cum"], color="#1f77b4")
        for _, row in top_dd.iterrows():
            # метки пиков и впадин
            pt = pd.to_datetime(row["peak_time"])
            tt = pd.to_datetime(row["trough_time"])
            ax.axvspan(pt, tt, color="#ff7f0e", alpha=0.12)
            ax.scatter([pt], [row["peak_val"]], color="#999999", s=20)
            ax.scatter([tt], [row["trough_val"]], color="#d62728", s=25)
        ax.set_title("Top Drawdowns (highlighted)")
        ax.set_xlabel("Time")
        ax.set_ylabel("PnL (RUB)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pnl_drawdowns.png"))
        plt.close(fig)

    # Сохранить график z‑score с порогами
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["dt"], df["z"], color="#9467bd", linewidth=1)
    if "z" in df.columns:
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.axhline(2.0, color="#d62728", linestyle="--", linewidth=0.8, label="z=+2")
        ax.axhline(-2.0, color="#2ca02c", linestyle="--", linewidth=0.8, label="z=-2")
    ax.set_title("Z-Score of Spread")
    ax.set_xlabel("Time")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "zscore.png"))
    plt.close(fig)

    # Сохранить график просадки
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df["dt"], df["drawdown"], color="#ff7f0e", linewidth=1)
    ax.set_title("Drawdown (RUB)")
    ax.set_xlabel("Time")
    ax.set_ylabel("DD")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "drawdown.png"))
    plt.close(fig)

    # Аналитика по корзинам волатильности спреда
    if "spread_vol" in df.columns and df["spread_vol"].notna().any():
        try:
            buckets = pd.qcut(df["spread_vol"].fillna(method="ffill"), 4, labels=False, duplicates="drop")
            df["vol_bucket"] = buckets
            grp = df.groupby("vol_bucket", dropna=True)
            ann_factor = np.sqrt(252 * 6.5 * 60)
            rows = []
            for b, g in grp:
                minutes = int(g.shape[0])
                pnl_sum = float(g["pnl_total"].sum())
                ret_mean = float(g["ret"].mean()) if "ret" in g.columns else float((g["pnl_total"]).mean())
                ret_std = float(g["ret"].std(ddof=0) + 1e-12) if "ret" in g.columns else float((g["pnl_total"]).std(ddof=0) + 1e-12)
                sharpe = (ret_mean / ret_std) * ann_factor if ret_std > 0 else 0.0
                rows.append({
                    "vol_bucket": int(b) if b == b else -1,
                    "minutes": minutes,
                    "spread_vol_med": float(g["spread_vol"].median()),
                    "pnl_sum": pnl_sum,
                    "ret_mean": ret_mean,
                    "ret_std": ret_std,
                    "sharpe": sharpe,
                })
            vol_stats = pd.DataFrame(rows).sort_values("vol_bucket")
            vol_stats.to_csv(os.path.join(out_dir, "stats_by_vol_bucket.csv"), index=False)

            # Визуализация эффективности по корзинам волатильности
            if not vol_stats.empty:
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax2 = ax1.twinx()
                x = vol_stats["vol_bucket"].astype(int)
                ax1.bar(x - 0.2, vol_stats["sharpe"], width=0.4, color="#1f77b4", label="Sharpe")
                ax2.bar(x + 0.2, vol_stats["pnl_sum"], width=0.4, color="#ff7f0e", label="PnL (RUB)")
                ax1.set_xlabel("Volatility bucket (quantiles)")
                ax1.set_ylabel("Sharpe")
                ax2.set_ylabel("PnL (RUB)")
                ax1.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "vol_buckets.png"))
                plt.close(fig)
        except Exception:
            # Безопасно пропустить, если qcut неустойчив на константном ряду
            pass

    # Аналитика по рыночным режимам
    if "regime" in df.columns:
        try:
            grp = df.groupby("regime", dropna=True)
            ann_factor = np.sqrt(252 * 6.5 * 60)
            rows = []
            for r, g in grp:
                minutes = int(g.shape[0])
                pnl_sum = float(g["pnl_total"].sum())
                ret_mean = float(g["ret"].mean()) if "ret" in g.columns else float((g["pnl_total"]).mean())
                ret_std = float(g["ret"].std(ddof=0) + 1e-12) if "ret" in g.columns else float((g["pnl_total"]).std(ddof=0) + 1e-12)
                sharpe_r = (ret_mean / ret_std) * ann_factor if ret_std > 0 else 0.0
                rows.append({
                    "regime": str(r),
                    "minutes": minutes,
                    "pnl_sum": pnl_sum,
                    "ret_mean": ret_mean,
                    "ret_std": ret_std,
                    "sharpe": sharpe_r,
                })
            reg_stats = pd.DataFrame(rows)
            reg_stats.to_csv(os.path.join(out_dir, "stats_by_regime.csv"), index=False)

            # Визуализация PnL с окраской по режимам
            try:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(df["dt"], df["pnl_cum"], color="#1f77b4")
                colors = {"bull": (0.2, 0.8, 0.2, 0.08), "bear": (0.8, 0.2, 0.2, 0.08), "sideways": (0.5, 0.5, 0.5, 0.06)}
                r_prev = None
                start = None
                for i, r in enumerate(df["regime"].astype(str)):
                    if r_prev is None:
                        r_prev = r; start = i
                    elif r != r_prev:
                        ax.axvspan(df["dt"].iloc[start], df["dt"].iloc[i-1], color=colors.get(r_prev, (0.5,0.5,0.5,0.05)))
                        r_prev = r; start = i
                if start is not None:
                    ax.axvspan(df["dt"].iloc[start], df["dt"].iloc[len(df)-1], color=colors.get(r_prev, (0.5,0.5,0.5,0.05)))
                ax.set_title("Cumulative PnL with Regime Shading")
                ax.set_xlabel("Time")
                ax.set_ylabel("PnL (RUB)")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "pnl_regimes.png"))
                plt.close(fig)
            except Exception:
                pass
        except Exception:
            pass


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    # Режим по умолчанию для обратной совместимости: использовать одиночную ОФЗ при наличии
    inputs = find_all_inputs(root)
    sber_path = inputs.get("SBER")
    ofz40_path = inputs.get("OFZ_26240")
    ofz41_path = inputs.get("OFZ_26241")
    rgbi_path = inputs.get("RGBI")

    config = StrategyConfig()
    df, stats = run_backtest(sber_path, ofz40_path, config, ofz41_path=ofz41_path, rgbi_path=rgbi_path)
    save_outputs(df, stats, os.path.join(root, "output"))

    # Вывести краткое резюме
    print("Backtest completed")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


