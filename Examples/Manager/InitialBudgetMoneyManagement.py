from MoneyManagement import MoneyManagement

class InitialBudgetMoneyManagement(MoneyManagement):
    __initial_cash: float

    def __init__(self, cash: float, max_leverage: int) -> None:
        super().__init__(cash, max_leverage)
        self.__initial_cash = cash

    def _calculate_lot(self, contract_size: float = 100000.0, current_price: float = 1.0, **kwargs) -> float:
        target_budget = min(self.__initial_cash, self.cash)

        if target_budget <= 0.0 or current_price <= 0.0 or contract_size <= 0.0:
            return 0.0

        raw_lot = target_budget / (contract_size * current_price)
        return raw_lot
