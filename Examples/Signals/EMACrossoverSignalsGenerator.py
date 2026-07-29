import numpy as np
from numpy import ndarray

from Signal import Signal
from SignalsGenerator import SignalsGenerator
from MoneyManagement import MoneyManagement
from DataIterator import DataIterator


class EMACrossoverSignalsGenerator(SignalsGenerator):
    __fast_period: int
    __slow_period: int
    __contract_size: float
    __invert_logic: bool

    __k_fast: float
    __k_slow: float

    __count: int
    __fast_ema_curr: float
    __fast_ema_prev: float
    __slow_ema_curr: float
    __slow_ema_prev: float
    __sum_fast_init: float
    __sum_slow_init: float

    def __init__(self, beat_form: MoneyManagement, iterator: DataIterator, fast_period: int = 10, slow_period: int = 30, contract_size: float = 100000.0, invert_logic: bool = False) -> None:
        super().__init__(beat_form, iterator)
        self.__fast_period = fast_period
        self.__slow_period = slow_period
        self.__contract_size = contract_size
        self.__invert_logic = invert_logic

        self.__k_fast = 2.0 / (fast_period + 1.0)
        self.__k_slow = 2.0 / (slow_period + 1.0)

        self.__count = 0
        self.__fast_ema_curr = None
        self.__fast_ema_prev = None
        self.__slow_ema_curr = None
        self.__slow_ema_prev = None

        self.__sum_fast_init = 0.0
        self.__sum_slow_init = 0.0

    def generate_signal(self, ohlc_bid: ndarray, spread: float) -> tuple[Signal, bytes]:
        close_price = float(ohlc_bid[3])
        self.__count += 1

        self.__fast_ema_prev = self.__fast_ema_curr
        self.__slow_ema_prev = self.__slow_ema_curr

        if self.__count <= self.__slow_period:
            if self.__count <= self.__fast_period:
                self.__sum_fast_init += close_price
                if self.__count == self.__fast_period:
                    self.__fast_ema_curr = self.__sum_fast_init / self.__fast_period

            self.__sum_slow_init += close_price
            if self.__count == self.__slow_period:
                self.__slow_ema_curr = self.__sum_slow_init / self.__slow_period

            return None, None

        self.__fast_ema_curr = (close_price * self.__k_fast) + (self.__fast_ema_curr * (1.0 - self.__k_fast))
        self.__slow_ema_curr = (close_price * self.__k_slow) + (self.__slow_ema_curr * (1.0 - self.__k_slow))

        if self.__fast_ema_prev is None or self.__slow_ema_prev is None:
            return None, None

        bullish_cross = (self.__fast_ema_prev <= self.__slow_ema_prev) and (self.__fast_ema_curr > self.__slow_ema_curr)
        bearish_cross = (self.__fast_ema_prev >= self.__slow_ema_prev) and (self.__fast_ema_curr < self.__slow_ema_curr)

        if self.__invert_logic:
            bullish_cross, bearish_cross = bearish_cross, bullish_cross

        active_signals = self._SignalsGenerator__signals
        open_ids = [s_id for s_id, sig in active_signals.items() if sig.open]

        if open_ids:
            current_trade_id = open_ids[0]
            current_trade = active_signals[current_trade_id]

            if (bullish_cross and not current_trade.long) or (bearish_cross and current_trade.long):
                current_trade.change_type()
                return current_trade, current_trade_id

        if not open_ids and (bullish_cross or bearish_cross):
            is_long = bullish_cross

            try:
                calculated_lot = self.management.lot(
                    contract_size=self.__contract_size,
                    current_price=close_price
                )
            except ValueError:
                return None, None

            if calculated_lot <= 0.0:
                return None, None

            new_signal = Signal(direction=is_long, lot=calculated_lot)
            signal_id = self.gen_id(new_signal)
            return new_signal, signal_id

        return None, None
