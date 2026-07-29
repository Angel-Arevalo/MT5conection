import random
from numpy import ndarray

from Signal import Signal
from SignalsGenerator import SignalsGenerator
from MoneyManagement import MoneyManagement
from DataIterator import DataIterator


class CoinFlipSignalsGenerator(SignalsGenerator):
    __contract_size: float

    def __init__(self, beat_form: MoneyManagement, iterator: DataIterator, contract_size: float = 100000.0) -> None:
        super().__init__(beat_form, iterator)
        self.__contract_size = contract_size

    def generate_signal(self, ohlc_bid: ndarray, spread: float) -> tuple[Signal, bytes]:
        close_price = float(ohlc_bid[3])

        action = random.choice(["ENTER", "EXIT", "NOTHING"])

        if action == "ENTER":
            is_long = random.choice([True, False])

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

        elif action == "EXIT":
            active_signals = self._SignalsGenerator__signals

            open_ids = [s_id for s_id, sig in active_signals.items() if sig.open]

            if not open_ids:
                return None, None

            chosen_id = random.choice(open_ids)
            signal_to_close = active_signals[chosen_id]

            signal_to_close.change_type()
            return signal_to_close, chosen_id

        return None, None
