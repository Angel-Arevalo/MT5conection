import pandas as pd
from numpy import ndarray

from pathlib import Path
import talib

from typing import Optional
from datetime import datetime, date, timedelta

from Signal import Signal
from SignalsGenerator import SignalsGenerator

from MoneyManagement import MoneyManagement
from DataIterator import DataIterator

FAST_METHODS: dict = {
    "SMA": talib.SMA, "EMA": talib.EMA, "WMA": talib.WMA,
    "DEMA": talib.DEMA, "TEMA": talib.TEMA, "TRIMA": talib.TRIMA,
    "KAMA": talib.KAMA, "T3": talib.T3, "MIDPOINT": talib.MIDPOINT,
}

class OptimalMA(SignalsGenerator):
    _params_long: Optional[dict]
    _params_short: Optional[dict]

    _current_monday: Optional[date]

    def __init__(self, asset_name: str, beat_form: MoneyManagement, iterator: DataIterator) -> None:
        super().__init__(beat_form, iterator)

        script_dir = Path(__file__).resolve().parent
        file_path = script_dir / "reversion_cache" / f"{asset_name}.csv"
        self._week_cache = self._load_week_cache(file_path)

        self._current_monday: Optional[date] = None
        self._params_long: Optional[dict] = None
        self._params_short: Optional[dict] = None
 
    def generate_signal(self, ohlc_bid: ndarray, spread: float) -> tuple[Signal, bytes]:
        week_changed = self._refresh_week_if_needed()

        if self._params_long is None and self._params_short is None:
            return None, None

        print(self._current_monday)
        print(self._params_short)
        print(self._params_long)

    def _monday_of(self, dt: datetime) -> date:
        return (dt - timedelta(days=dt.weekday())).date()

    def _refresh_week_if_needed(self) -> bool:
        last_updt = self._iterator.market.last_updt
        if last_updt is None:
            return False

        monday = self._monday_of(last_updt)
        if monday == self._current_monday:
            return False

        self._current_monday = monday
        vals = self._week_cache.get(monday)

        if vals is None:
            self._params_long = self._params_short = None
            return True

        self._params_long  = {'met': vals['met_l'], 'vela': vals['vela_l'], 'lb': vals['lb_l']}
        self._params_short = {'met': vals['met_s'], 'vela': vals['vela_s'], 'lb': vals['lb_s']}

        return True

    def _load_week_cache(self, file_path) -> dict[date, dict]:
        df = pd.read_csv(file_path, dtype={'semana': str, 'met_l': str, 'met_s': str})
        cache = {}
        for _, row in df.iterrows():
            monday = date.fromisoformat(row['semana'])
            cache[monday] = {
                'met_l': row['met_l'], 'vela_l': int(row['vela_l']), 'lb_l': int(row['lb_l']),
                'met_s': row['met_s'], 'vela_s': int(row['vela_s']), 'lb_s': int(row['lb_s']),
            }
        return cache
