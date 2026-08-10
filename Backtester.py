from collections import defaultdict
from typing import Optional
import pandas as pd

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
    __open_signals: dict[bytes, dict]

    __raw_trades: dict[SignalsGenerator, list[dict]]
    __closed_trades: dict[SignalsGenerator, pd.DataFrame]

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

        self.__open_signals = {}
        self.__raw_trades = defaultdict(list)
        self.__closed_trades = {}

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

    def _run_simulation(self) -> None:
        data, spread = self.__iterator.next_candle()

        while data.size > 0:
            current_time = self.__iterator.market.last_updt
            close_p = data[3]

            for model in self.__models:
                signals = model.generate_signal(data, spread)

                if not signals:
                    continue

                for signal, sid in signals:
                    if signal is None:
                        continue

                    if signal.open:
                        entry_price = close_p + spread if signal.long else close_p

                        self.__open_signals[sid] = {
                            "model": model,
                            "entry_price": entry_price,
                            "signal": signal,
                            "open_time": current_time,
                        }
                    else:
                        if sid not in self.__open_signals:
                            continue

                        trade = self.__open_signals.pop(sid)
                        open_signal = trade["signal"]
                        entry_price = trade["entry_price"]

                        exit_price = close_p if open_signal.long else close_p + spread

                        self.__raw_trades[model].append({
                            "is_long": open_signal.long,
                            "open_time": trade["open_time"],
                            "close_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "lot": open_signal.lot,
                        })

                        model.delete_signal(sid)

            data, spread = self.__iterator.next_candle()

        mt5.shutdown()

    def _compute_trade_financials(self, model: SignalsGenerator) -> pd.DataFrame:
        raw = self.__raw_trades.get(model, [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)

        diff = np.where(
            df["is_long"],
            df["exit_price"] - df["entry_price"],
            df["entry_price"] - df["exit_price"],
        )

        df["gross_pnl"] = diff * df["lot"] * self.__contract_size
        df["commission"] = df["lot"] * self.__commission_per_lot

        df["swap"] = [
            self._calculate_swap(
                is_long=row.is_long,
                lot=row.lot,
                open_time=row.open_time,
                close_time=row.close_time,
                entry_price=row.entry_price,
                exit_price=row.exit_price,
            )
            for row in df.itertuples(index=False)
        ]

        df["pnl"] = df["gross_pnl"] - df["commission"] + df["swap"]

        df["duration_hours"] = df.apply(
            lambda r: (r["close_time"] - r["open_time"]).total_seconds() / 3600.0
            if r["open_time"] and r["close_time"] else 0.0,
            axis=1,
        )

        total_pnl = float(df["pnl"].sum())
        model.management.cash += total_pnl

        self.__closed_trades[model] = df
        return df

    def start(self) -> dict[str, float]:
        self._run_simulation()

        for model in self.__models:
            self._compute_trade_financials(model)

        return {
            f"model_{i}_cash": model.management.cash
            for i, model in enumerate(self.__models)
        }

    def get_trades_df(self, model: SignalsGenerator) -> pd.DataFrame:
        df = self.__closed_trades.get(model)
        if df is None:
            return pd.DataFrame()
        return df

    def get_metrics(self, model: SignalsGenerator) -> dict:
        df = self.get_trades_df(model)

        if df.empty:
            return {
                "total_trades": 0,
                "final_cash": round(model.management.cash, 2)
            }

        total_trades = len(df)
        winning_trades = df[df["pnl"] > 0]
        losing_trades = df[df["pnl"] < 0]

        win_rate = (len(winning_trades) / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = float(winning_trades["pnl"].sum())
        gross_loss = abs(float(losing_trades["pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

        df = df.sort_values("close_time")
        df["cumulative_pnl"] = df["pnl"].cumsum()
        running_max = np.maximum.accumulate(df["cumulative_pnl"])
        drawdown = running_max - df["cumulative_pnl"]
        max_drawdown = float(drawdown.max())

        return {
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "net_pnl": round(float(df["pnl"].sum()), 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2) if not np.isnan(profit_factor) else "N/A",
            "max_drawdown_usd": round(max_drawdown, 2),
            "total_commissions": round(float(df["commission"].sum()), 2),
            "total_swaps": round(float(df["swap"].sum()), 2),
            "avg_trade_duration_hrs": round(float(df["duration_hours"].mean()), 2),
            "final_cash": round(model.management.cash, 2)
        }

    def add_model(self, new_model: SignalsGenerator) -> None:
        self.__models.append(new_model)

    @property
    def min_lot(self) -> float:
        return self.__min_lot

    @property
    def lot_step(self) -> float:
        return self.__lot_step

    @property
    def iterator(self) -> DataIterator:
        return self.__iterator
