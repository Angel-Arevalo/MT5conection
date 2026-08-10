from MoneyManagement import MoneyManagement
from typing import Any


class InitialBudgetMoneyManagement(MoneyManagement):
    __initial_cash: float

    def __init__(self, cash: float, max_leverage: int, symbol_info: Any) -> None:
        super().__init__(cash, max_leverage, symbol_info)
        self.__initial_cash = cash

    def _calculate_lot(self, current_price: float, contract_size: float, **kwargs) -> float:
        target_budget = min(self.__initial_cash, self.cash)

        if target_budget <= 0.0 or current_price <= 0.0 or contract_size <= 0.0:
            return 0.0

        raw_lot = target_budget / (contract_size * current_price)
        return raw_lot
