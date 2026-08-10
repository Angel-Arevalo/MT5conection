import pandas as pd
from numpy import ndarray
import sys

from pathlib import Path
import talib

from typing import Optional
from datetime import datetime, date, timedelta, timezone, time as dtime

from Signal import Signal
from SignalsGenerator import SignalsGenerator

from MoneyManagement import MoneyManagement
from DataIterator import DataIterator

submodule_path = Path(__file__).resolve().parent / "optimal-moving-average"
sys.path.append(str(submodule_path))

import keys
from find_best import opti_main

keys.calls = 10

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
        self._file_path = script_dir / "reversion_cache" / f"{asset_name}.csv"
        self._week_cache = self._load_week_cache(self._file_path)

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
        self._opt_week(self._current_monday)


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

    def _opt_week(self, monday_week: datetime) -> None:
        monday_start = datetime.combine(
            monday_week - timedelta(weeks=4), 
            dtime.min, 
            tzinfo=timezone.utc
        )

        fryday_end = datetime.combine(
            monday_week - timedelta(days=3), 
            dtime.max, 
            tzinfo=timezone.utc
        )

        data_opt: pd.DataFrame = self._iterator.sub_data(monday_start, fryday_end)
        data_opt.rename(columns={'close': 'bid'}, inplace=True)
        data_opt["ask"] = data_opt["bid"] + data_opt["spread"]

        data_opt["Precio Spot"] = (data_opt["ask"] + data_opt["bid"])/2
        data_opt.drop(columns=['open', 'high', 'low', 'spread'], inplace=True)

        met_l, cand_l, lo_l = opti_main(data_opt, True, True, "fm", False)
        met_s, cand_s, lo_s = opti_main(data_opt, True, True, "fm", False)

        self._week_cache[monday_week] = {
            'met_l': met_l, 'vela_l': int(cand_l), 'lb_l': int(lo_l),
            'met_s': met_s, 'vela_s': int(cand_s), 'lb_s': int(lo_s)
        }

        self._params_long = {'met': met_l, 'vela': int(cand_l), 'lb': int(lo_l)}
        self._params_short = {'met': met_s, 'vela': int(cand_s), 'lb': int(lo_s)}

        self._save_week_cache()

    def _save_week_cache(self) -> None:
        rows = []

        for monday_date, vals in sorted(self._week_cache.items()):
            rows.append({
                'semana': monday_date.isoformat(),
                'met_l': vals['met_l'],
                'vela_l': vals['vela_l'],
                'lb_l': vals['lb_l'],
                'met_s': vals['met_s'],
                'vela_s': vals['vela_s'],
                'lb_s': vals['lb_s'],
            })
 
        df = pd.DataFrame(rows)
        df.to_csv(self._file_path, index=False)
