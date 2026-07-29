from datetime import datetime
from Backtester import Backtester
from DataIterator import DataIterator

from Examples.Manager.InitialBudgetMoneyManagement import InitialBudgetMoneyManagement
from Examples.Signals.CoinFlipSignalsGenerator import CoinFlipSignalsGenerator
from Examples.Signals.EMACrossoverSignalsGenerator import EMACrossoverSignalsGenerator

def main():
    asset = "EURUSD_"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    iterator = DataIterator(asset)
    bt = Backtester(asset, start_date, end_date)

    mm = InitialBudgetMoneyManagement(cash=100000.0, max_leverage=100)
    model = EMACrossoverSignalsGenerator(
            beat_form=mm, 
            iterator=iterator, 
            fast_period=10, 
            slow_period=30, 
            contract_size=100000.0,
            invert_logic=True
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
    main()
