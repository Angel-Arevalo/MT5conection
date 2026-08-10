from datetime import datetime
import MetaTrader5 as mt5

from Backtester import Backtester
from DataIterator import DataIterator

from Examples.Manager.InitialBudgetMoneyManagement import InitialBudgetMoneyManagement
from Examples.Signals.CoinFlipSignalsGenerator import CoinFlipSignalsGenerator
from Examples.Signals.EMACrossoverSignalsGenerator import EMACrossoverSignalsGenerator
from Examples.Signals.OptimalMA import OptimalMA

symbol: str = "AUDCAD_"

if mt5.initialize():
    info = mt5.symbol_info(symbol)

    mt5.shutdown()

managament = InitialBudgetMoneyManagement(1000, 200, info)
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

iterator = DataIterator(symbol)
iterator.backtest(start_date, end_date)
#iterator.next_candle()

strategi = OptimalMA(symbol, managament, iterator)

strategi.generate_signal([1, 1, 1, 1], 1)


"""
def main():
    asset = "EURUSD_"
    bt = Backtester(asset, start_date, end_date)

    mm = InitialBudgetMoneyManagement(cash=100000.0, max_leverage=100)
    model = CoinFlipSignalsGenerator(
            beat_form=mm, 
            iterator=iterator, 
            contract_size=100000.0,
    )

    bt.add_model(model)

    print("Ejecutando Backtest...")
    bt.start()

    kpis = bt.get_metrics(model)
    print("\n--- RESUMEN DE KPIs ---")
    for k, v in kpis.items():
        print(f"{k}: {v}")

    df_trades = bt.get_trades_df(model)
    print("\n--- DETALLE DE OPERACIONES (primeras 5) ---")
    print(df_trades.head())

if __name__ == "__main__":
    main()"""
