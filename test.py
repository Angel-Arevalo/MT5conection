from datetime import datetime, timedelta
import MetaTrader5 as mt5

from Backtester import Backtester
from DataIterator import DataIterator

from Examples.Manager.InitialBudgetMoneyManagement import InitialBudgetMoneyManagement
from Examples.Signals.CoinFlipSignalsGenerator import CoinFlipSignalsGenerator
from Examples.Signals.EMACrossoverSignalsGenerator import EMACrossoverSignalsGenerator
from Examples.Signals.OptimalMA import OptimalMA


def main():
    asset = "AUDCAD_"
    start_date = datetime(2024, 12, 9)
    end_date = datetime(2025, 12, 31)

    bt = Backtester(asset, start_date, end_date)
    symbol_info = mt5.symbol_info(asset)

    mm_mr = InitialBudgetMoneyManagement(cash=50_000.0, max_leverage=100, symbol_info=symbol_info)
    model_mr = OptimalMA(asset_name=asset, beat_form=mm_mr, iterator=bt.iterator)

    mm_ema = InitialBudgetMoneyManagement(cash=50_000.0, max_leverage=100, symbol_info=symbol_info)
    model_ema = EMACrossoverSignalsGenerator(
        beat_form=mm_ema, iterator=bt.iterator,
        fast_period=10, slow_period=30, invert_logic=True,
    )

    bt.add_model(model_mr)
    bt.add_model(model_ema)
    print("Ejecutando Backtest...")
    bt.start()

    for name, model in (("OptimalMA", model_mr), ("EMACrossover", model_ema)):
        kpis = bt.get_metrics(model)
        print(f"\n--- RESUMEN DE KPIs ({name}) ---")
        for k, v in kpis.items():
            print(f"{k}: {v}")

        df_trades = bt.get_trades_df(model)
        print(f"\n--- DETALLE DE OPERACIONES ({name}, primeras 5) ---")
        print(df_trades.head())


if __name__ == "__main__":
    main()
