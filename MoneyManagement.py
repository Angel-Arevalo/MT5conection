from abc import ABC, abstractmethod
from typing import Any
import MetaTrader5 as mt5


class MoneyManagement(ABC):
    __cash: float
    __max_leverage: int
    _symbol_info: Any

    def __init__(self, cash: float, max_leverage: int, symbol_info: Any) -> None:
        self.__cash = cash
        self.__max_leverage = max_leverage
        self._symbol_info = symbol_info

    @abstractmethod
    def _calculate_lot(self, current_price: float, contract_size: float, **kwargs) -> float:
        pass

    def lot(self, current_price: float, order_type: int = mt5.ORDER_TYPE_BUY, **kwargs) -> float:
        min_lot: float = self._symbol_info.volume_min
        lot_step: float = self._symbol_info.volume_step
        contract_size: float = self._symbol_info.trade_contract_size

        raw_lot: float = self._calculate_lot(
            current_price=current_price, contract_size=contract_size, **kwargs
        )

        final_lot: float = self.normalize_lot(raw_lot, min_lot, lot_step)

        if final_lot <= 0.0:
            return 0.0

        if not self._has_enough_margin(final_lot, current_price, order_type):
            return 0.0

        return final_lot

    def _has_enough_margin(self, lot: float, current_price: float, order_type: int) -> bool:
        account = mt5.account_info()
        if account is None:
            return False

        required_margin = mt5.order_calc_margin(
            order_type,
            self._symbol_info.name,
            lot,
            current_price,
        )

        if required_margin is None:
            return False

        margin_free = account.margin_free

        max_allowed_by_config = self.__cash * self.__max_leverage
        exposure = lot * self._symbol_info.trade_contract_size * current_price

        if exposure > max_allowed_by_config:
            return False

        return required_margin <= margin_free

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

    @property
    def info(self) -> Any:
        return self._symbol_info
