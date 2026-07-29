from abc import ABC, abstractmethod

class MoneyManagement(ABC):
    __cash: float
    __max_leverage: int

    def __init__(self, cash: float, max_leverage: int) -> None:
        self.__cash = cash
        self.__max_leverage = max_leverage

    @abstractmethod
    def _calculate_lot(self, **kwargs) -> float:
        pass

    def lot(self, contract_size: float, current_price: float, min_lot: float = 0.01, lot_step: float = 0.01, **kwargs) -> float:
        raw_lot: float = self._calculate_lot(**kwargs)

        final_lot: float = self.normalize_lot(raw_lot, min_lot, lot_step)

        if final_lot <= 0.0:
            return 0.0

        exposure: float = final_lot * contract_size * current_price
        max_allowed_exposure: float = self.__cash * self.__max_leverage

        if exposure > max_allowed_exposure:
            raise ValueError(f"Sin margen suficiente. Exposición requerida: {exposure}, Máxima permitida: {max_allowed_exposure}")

        return final_lot

    def normalize_lot(self, raw_lot: float, min_lot: float = 0.01, lot_step: float = 0.01) -> float:
        if raw_lot < min_lot:
            return 0.0

        steps = round(raw_lot / lot_step)
        return round(steps * lot_step, 2)

    @property
    def cash(self) -> float:
        return self.__cash

    @cash.setter
    def cash(self, value: float) -> None:
        self.__cash = value
