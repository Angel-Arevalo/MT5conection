import math
from collections import defaultdict

from typing import Optional
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np

from DataIterator import DataIterator
from MarketStatus import MarketStatus

from MoneyManagement import MoneyManagement
from Signal import Signal
from SignalsGenerator import SignalsGenerator


class Backtester:

    __name: str
    __iterator: DataIterator

    __models: list[SignalsGenerator]
    __signals: dict[bytes, dict]

    __closed_trades: dict[SignalsGenerator, list[dict]]
    __contract_size: float

    def __init__(self, asset_name: str, start_day: datetime, end_day: datetime) -> None:
        self.__name = asset_name
        self.__iterator = DataIterator(asset_name)
        self.__iterator.backtest(start_day, end_day)

        self.__models = []
        self.__signals = {}
        self.__closed_trades = defaultdict(list)

        if not mt5.initialize():
            raise RuntimeError(f"Error al inicializar MT5: {mt5.last_error()}")

        symbol_info: Optional[mt5.SymbolInfo] = mt5.symbol_info(self.__name)
        if symbol_info is None:
            mt5.shutdown()
            raise ValueError(f"Símbolo {self.__name} no encontrado en MT5.")

        self.__contract_size = float(symbol_info.trade_contract_size)

    def start(self) -> dict[str, float]:
        data, spread = self.__iterator.next_candle()

        while data.size > 0:

            current_time = self.__iterator.market.last_updt

            high_p = data[1]
            low_p = data[2]
            close_p = data[3]

            for trade_id, trade in self.__signals.items():
                trade["max_high"] = max(trade["max_high"], high_p)
                trade["min_low"] = min(trade["min_low"], low_p)

            for model in self.__models:
                signal, id = model.generate_signal(data, spread)

                if signal is not None:
                    if signal.open:
                        entry_price = (
                            close_p + spread if signal.long else close_p
                        )

                        self.__signals[id] = {
                            "entry_price": entry_price,
                            "signal": signal,
                            "open_time": current_time,
                            "max_high": high_p,
                            "min_low": low_p,
                        }
                    else:
                        if id in self.__signals:
                            trade = self.__signals.pop(id)
                            open_signal = trade["signal"]
                            entry_price = trade["entry_price"]

                            if open_signal.long:
                                exit_price = close_p
                                diff = exit_price - entry_price

                                mfe_p = trade["max_high"] - entry_price
                                mae_p = entry_price - trade["min_low"]

                            else:
                                exit_price = close_p + spread
                                diff = entry_price - exit_price

                                mfe_p = entry_price - trade["min_low"]
                                mae_p = trade["max_high"] - entry_price

                            pnl = diff * open_signal.lot * self.__contract_size

                            mae_usd = max(
                                0.0,
                                mae_p * open_signal.lot * self.__contract_size,
                            )

                            mfe_usd = max(
                                0.0,
                                mfe_p * open_signal.lot * self.__contract_size,
                            )

                            duration_hrs = (
                                (current_time - trade["open_time"]).total_seconds() / 3600.0
                                if trade["open_time"] and current_time
                                else 0.0
                            )

                            model.management.cash += pnl
                            model.delete_signal(id)

                            self.__closed_trades[model].append(
                                {
                                    "is_long": open_signal.long,
                                    "open_time": trade["open_time"],
                                    "close_time": current_time,
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "lot": open_signal.lot,
                                    "pnl": pnl,
                                    "mae_usd": mae_usd,
                                    "mfe_usd": mfe_usd,
                                    "duration_hours": duration_hrs,
                                    "swap": 0.0,
                                    "commission": 0.0,
                                }
                            )

            data, spread = self.__iterator.next_candle()

        mt5.shutdown()

        return {
            f"model_{i}_cash": model.management.cash
            for i, model in enumerate(self.__models)
        }

    def add_model(self, new_model: SignalsGenerator) -> None:
        self.__models.append(new_model)
