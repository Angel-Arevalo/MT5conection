from collections import defaultdict
from typing import Optional

from datetime import datetime, timedelta
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
    __min_lot: float

    __lot_step: float
    __point: float

    __swap_mode: int
    __swap_long: float
    __swap_short: float
    __swap_3day_py_weekday: int
    __commission_per_lot: float

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
        self.__min_lot = float(symbol_info.volume_min)
        self.__lot_step = float(symbol_info.volume_step)
        self.__point = float(symbol_info.point)

        self.__swap_mode = int(symbol_info.swap_mode)
        self.__swap_long = float(symbol_info.swap_long)
        self.__swap_short = float(symbol_info.swap_short)

        mt5_3day = int(symbol_info.swap_rollover3days)
        self.__swap_3day_py_weekday = (mt5_3day - 1) % 7

        self.__commission_per_lot = self._get_commission_from_history()

    def _get_commission_from_history(self) -> float:
        deals = mt5.history_deals_get(datetime(2020, 1, 1), datetime.now())
        if deals:
            for deal in deals:
                if deal.symbol == self.__name and deal.volume > 0 and deal.commission != 0:
                    return abs(deal.commission) / deal.volume
        return 0.0009

    def _calculate_daily_swap_usd(self, is_long: bool, lot: float, entry_price: float, current_price: float) -> float:
        raw_swap = self.__swap_long if is_long else self.__swap_short

        if raw_swap == 0.0 or self.__swap_mode == mt5.SYMBOL_SWAP_MODE_DISABLED:
            return 0.0

        mode = self.__swap_mode

        if mode == mt5.SYMBOL_SWAP_MODE_POINTS:
            return raw_swap * self.__point * self.__contract_size * lot

        elif mode == mt5.SYMBOL_SWAP_MODE_CURRENCY_SYMBOL:
            return raw_swap * lot * current_price

        elif mode == mt5.SYMBOL_SWAP_MODE_CURRENCY_MARGIN:
            return raw_swap * lot * current_price

        elif mode == mt5.SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT:
            return raw_swap * lot

        elif mode == mt5.SYMBOL_SWAP_MODE_INTEREST_CURRENT:
            return (current_price * self.__contract_size * lot * (raw_swap / 100.0)) / 360.0

        elif mode == mt5.SYMBOL_SWAP_MODE_INTEREST_OPEN:
            return (entry_price * self.__contract_size * lot * (raw_swap / 100.0)) / 360.0

        elif mode in (mt5.SYMBOL_SWAP_MODE_REOPEN_BY_CLOSE_PRICE, mt5.SYMBOL_SWAP_MODE_REOPEN_BY_BID):
            return raw_swap * self.__point * self.__contract_size * lot

        elif mode == mt5.SYMBOL_SWAP_MODE_CURRENCY_PROFIT:
            return raw_swap * lot

        return raw_swap * self.__point * self.__contract_size * lot

    def _calculate_swap(self, is_long: bool, lot: float, open_time: datetime, close_time: datetime, entry_price: float, exit_price: float) -> float:
        if not open_time or not close_time or open_time >= close_time:
            return 0.0

        if self.__swap_mode == mt5.SYMBOL_SWAP_MODE_DISABLED:
            return 0.0

        daily_swap_usd = self._calculate_daily_swap_usd(
            is_long=is_long,
            lot=lot,
            entry_price=entry_price,
            current_price=exit_price
        )

        current_check = (open_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        total_swap = 0.0

        while current_check <= close_time:
            rollover_weekday = (current_check - timedelta(days=1)).weekday()

            if rollover_weekday == self.__swap_3day_py_weekday:
                total_swap += daily_swap_usd * 3.0

            else:
                total_swap += daily_swap_usd * 1.0

            current_check += timedelta(days=1)

        return total_swap

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

                            gross_pnl = diff * open_signal.lot * self.__contract_size
                            commission = open_signal.lot * self.__commission_per_lot

                            swap = self._calculate_swap(
                                is_long=open_signal.long,
                                lot=open_signal.lot,
                                open_time=trade["open_time"],
                                close_time=current_time,
                                entry_price=entry_price,
                                exit_price=exit_price
                            )

                            net_pnl = gross_pnl - commission + swap

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

                            model.management.cash += net_pnl
                            model.delete_signal(id)

                            self.__closed_trades[model].append(
                                {
                                    "is_long": open_signal.long,
                                    "open_time": trade["open_time"],
                                    "close_time": current_time,
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "lot": open_signal.lot,
                                    "pnl": net_pnl,
                                    "gross_pnl": gross_pnl,
                                    "mae_usd": mae_usd,
                                    "mfe_usd": mfe_usd,
                                    "duration_hours": duration_hrs,
                                    "swap": swap,
                                    "commission": commission,
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

    @property
    def min_lot(self) -> float:
        return self.__min_lot

    @property
    def lot_step(self) -> float:
        return self.__lot_step
