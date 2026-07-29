import datetime
import MetaTrader5 as mt5
from typing import Optional

from DataIterator import DataIterator
from MarketStatus import MarketStatus

from MoneyManagement import MoneyManagement
from Signal import Signal

from SignalsGenerator import SignalsGenerator


class Backtester:

    __name: str
    __iterator: DataIterator

    __models: list[SignalsGenerator]
    __signals: dict[bytes, tuple[float, Signal]]

    __contract_size: float

    def __init__(self, asset_name: str, start_day: datetime.datetime, end_day: datetime.datetime) -> None:
        self.__name = asset_name
        self.__iterator = DataIterator(asset_name)
        self.__iterator.backtest(start_day, end_day)

        self.__models = []
        self.__signals = {}

        if not mt5.initialize():
            raise RuntimeError(f"Error al inicializar MT5: {mt5.last_error()}")

        symbol_info: Optional[mt5.SymbolInfo] = mt5.symbol_info(self.__name)
        if symbol_info is None:
            raise ValueError(f"Símbolo {self.__name} no encontrado en MT5.")

        self.__contract_size = float(symbol_info.trade_contract_size)


    def start(self) -> dict[str, float]:
        data, spread = self.__iterator.next_candle()

        while len(data) == 4:
            for model in self.__models:
                signal, id = model.generate_signal(data, spread)

                if signal is not None:
                    if signal.open:
                        if signal.long:

                            self.__signals[id] = (data[3] + spread, signal)
                        else:

                            self.__signals[id] = (data[3], signal)

                    else:
                        if id in self.__signals:
                            entry_price, open_signal = self.__signals.pop(id)

                            if open_signal.long:
                                exit_price = data[3]
                                diff = exit_price - entry_price

                            else:
                                exit_price = data[3] + spread
                                diff = entry_price - exit_price

                            diff = diff * open_signal.lot * self.__contract_size
                            model.management.cash += diff

                            model.delete_signal(id)


            data, spread = self.__iterator.next_candle()


    def add_model(self, new_model: SignalsGenerator) -> None:
        self.__models.append(new_model)
