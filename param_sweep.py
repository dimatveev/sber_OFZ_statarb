import os
import itertools
import pandas as pd

from arb_backtest import StrategyConfig, run_backtest, find_input_files


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    sber_path, ofz_path = find_input_files(root)

    # Поисковая сетка параметров
    z_lookbacks = [120, 180, 240, 360]
    z_enters = [1.5, 2.0, 2.5]
    z_exits = [0.25, 0.5, 0.75]

    results = []
    for zlb, ze, zx in itertools.product(z_lookbacks, z_enters, z_exits):
        config = StrategyConfig(
            z_lookback_min=zlb,
            z_enter=ze,
            z_exit=zx,
        )
        df, stats = run_backtest(sber_path, ofz_path, config)
        results.append({
            "z_lookback_min": zlb,
            "z_enter": ze,
            "z_exit": zx,
            **stats,
        })

    res_df = pd.DataFrame(results)
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "param_sweep.csv")
    res_df.to_csv(out_csv, index=False)

    # Вывести топ-10 по коэффициенту Шарпа
    top = res_df.sort_values("sharpe", ascending=False).head(10)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()


