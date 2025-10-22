import os
from typing import Tuple

from arb_backtest import StrategyConfig, run_backtest, find_input_files, save_outputs, find_all_inputs


BEST_SHARPE = dict(z_lookback_min=180, z_enter=1.5, z_exit=0.75)
BEST_PNL = dict(z_lookback_min=120, z_enter=1.5, z_exit=0.25)


def run_case(root: str, label: str, timeframe_label: str, timeframe_cfg: str, params: dict, hedge_mode: str = "OFZ_SINGLE", ofz_weights=(0.5, 0.5)) -> Tuple[str, dict]:
	"""Запустить один сценарий и сохранить результаты в output/{label}_{timeframe_label}_{hedge}"""
	inputs = find_all_inputs(root)
	sber_path = inputs.get("SBER")
	ofz40_path = inputs.get("OFZ_26240")
	ofz41_path = inputs.get("OFZ_26241")
	rgbi_path = inputs.get("RGBI")
	cfg = StrategyConfig(
		z_lookback_min=params["z_lookback_min"],
		z_enter=params["z_enter"],
		z_exit=params["z_exit"],
		timeframe=timeframe_cfg,
		hedge_mode=hedge_mode,
		ofz_weights=ofz_weights,
	)
	df, stats = run_backtest(sber_path, ofz40_path, cfg, ofz41_path=ofz41_path, rgbi_path=rgbi_path)
	hedge_label = hedge_mode.lower()
	out_dir = os.path.join(root, "output", f"{label}_{timeframe_label}_{hedge_label}")
	save_outputs(df, stats, out_dir)
	return out_dir, stats


def main():
	root = os.path.dirname(os.path.abspath(__file__))
	cases = [
		# Базовый сценарий: одиночная ОФЗ
		("best_sharpe", "1min", None, BEST_SHARPE, "OFZ_SINGLE"),
		("best_pnl",    "1min", None, BEST_PNL,    "OFZ_SINGLE"),
		("best_sharpe", "5min", "5min", BEST_SHARPE, "OFZ_SINGLE"),
		("best_pnl",    "5min", "5min", BEST_PNL,    "OFZ_SINGLE"),
		# Хедж через индекс RGBI
		("best_sharpe", "1min", None, BEST_SHARPE, "RGBI"),
		("best_pnl",    "1min", None, BEST_PNL,    "RGBI"),
		("best_sharpe", "5min", "5min", BEST_SHARPE, "RGBI"),
		("best_pnl",    "5min", "5min", BEST_PNL,    "RGBI"),
		# Корзина ОФЗ 26240/26241 в пропорции 50/50
		("best_sharpe", "1min", None, BEST_SHARPE, "OFZ_BASKET"),
		("best_pnl",    "1min", None, BEST_PNL,    "OFZ_BASKET"),
		("best_sharpe", "5min", "5min", BEST_SHARPE, "OFZ_BASKET"),
		("best_pnl",    "5min", "5min", BEST_PNL,    "OFZ_BASKET"),
	]
	for label, tf_label, tf_cfg, params, hedge in cases:
		out_dir, stats = run_case(root, label, tf_label, tf_cfg, params, hedge_mode=hedge)
		print(f"{label} {tf_label} {hedge} -> {out_dir} :: {stats}")


if __name__ == "__main__":
	main()
