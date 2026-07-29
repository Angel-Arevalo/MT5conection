from abc import ABC, abstractmethod

class MoneyManagement(ABC):
    __cash: float

    __max_leverage: int

    def __init__(self, cash: float, max_leverage: int) -> None:
        self.__cash = cash
        self.__max_leverage = max_leverage

    @abstractmethod
    def calculate_lot(self, **kwargs) -> float:
        pass

    def normalize_lot(self, raw_lot: float, min_lot: float = 0.01, lot_step: float = 0.01) -> float:
        if raw_lot < min_lot:
            return 0.0

        steps = round(raw_lot / lot_step)
        return round(steps * lot_step, 2)

    @property
    def cash(self) -> float:
        return self._cash

    @cash.setter
    def cash(self, value: float) -> None:
        self.__cash = value
