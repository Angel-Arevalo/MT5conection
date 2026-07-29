from abc import ABC, abstractmethod
from numpy import ndarray

from secrets import token_bytes
from Signal import Signal
from MoneyManagement import MoneyManagement

class SignalsGenerator(ABC):

    __signals: dict[bytes, Signal]
    __management: MoneyManagement

    def __init__(self, beat_form: MoneyManagement) -> None:
        self.__signals = {}
        self.__management = beat_form

    @abstractmethod
    def generate_signal(self, ohlc_bid: ndarray, spread: float) -> tuple[Signal, bytes]:
        pass

    def gen_id(self, signal: Signal) -> bytes:
        key: bytes = token_bytes(8)
        self.__signals[key] = signal
        return key

    def delete_signal(self, id):
        del self.__signals[id]

    @property
    def management(self) -> MoneyManagement:
        return self.__management
