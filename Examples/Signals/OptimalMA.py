import numpy as np
import pandas as pd
from numpy import ndarray
import sys

from pathlib import Path
import talib
import MetaTrader5 as mt5

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

        self._iterator = iterator

        script_dir = Path(__file__).resolve().parent
        self._file_path = script_dir / "reversion_cache" / f"{asset_name}.csv"
        self._week_cache = self._load_week_cache(self._file_path)

        self._current_monday: Optional[date] = None
        self._params_long: Optional[dict] = None
        self._params_short: Optional[dict] = None

        self._minute_count: dict[bool, int] = {True: 0, False: 0}

        self._positions_long: dict[bytes, dict] = {}
        self._positions_short: dict[bytes, dict] = {}

        self._last_weekend_close_ts: Optional[int] = None

    def generate_signal(self, ohlc_bid: ndarray, spread: float) -> list[tuple[Signal, bytes]]:
        results: list[tuple[Signal, bytes]] = []

        forced = self._check_weekend_close()
        if forced:
            results += forced
            self._refresh_week_if_needed()
            return results

        self._refresh_week_if_needed()

        if self._params_long is None or self._params_short is None:
            return results

        self._minute_count[True] += 1
        self._minute_count[False] += 1

        results += self._eval_side(
            es_long=True, vela=self._params_long['vela'],
            lb=self._params_long['lb'], metodo=self._params_long['met'],
        )
        results += self._eval_side(
            es_long=False, vela=self._params_short['vela'],
            lb=self._params_short['lb'], metodo=self._params_short['met'],
        )

        return results

    def _eval_side(self, es_long: bool, vela: int, lb: int, metodo: str) -> list[tuple[Signal, bytes]]:
        out: list[tuple[Signal, bytes]] = []

        if self._minute_count[es_long] % vela != 0:
            return out

        needed_minutes = (lb + 25) * vela
        w = self._iterator.last_candles(needed_minutes)
        if len(w) < 2:
            return out

        w_sub = w.iloc[::-1].iloc[::vela].iloc[::-1]
        if len(w_sub) < lb + 5:
            return out

        arr = w_sub['close'].to_numpy(dtype=float)
        ask = arr + w_sub['spread'].to_numpy(dtype=float)

        bid_ = float(arr[-1])
        ask_ = float(ask[-1])

        ma = FAST_METHODS[metodo](arr, timeperiod=lb)
        if len(ma) < 2 or np.isnan(ma[-1]) or np.isnan(ma[-2]):
            return out

        if arr[-2] >= ma[-2] and arr[-1] < ma[-1]:
            s = 1
        elif arr[-2] <= ma[-2] and arr[-1] > ma[-1]:
            s = -1
        else:
            s = 0

        target_entrada = 1 if es_long else -1
        target_salida = -1 if es_long else 1
        positions = self._positions_long if es_long else self._positions_short

        if s == target_salida and positions:
            for sid, pos in list(positions.items()):
                pos['signal'].change_type()
                out.append((pos['signal'], sid))
                del positions[sid]

        if s == target_entrada:
            precio_in = ask_ if es_long else bid_
            order_type = mt5.ORDER_TYPE_BUY if es_long else mt5.ORDER_TYPE_SELL

            lot = self.management.lot(
                current_price=precio_in,
                order_type=order_type,
                es_long=es_long, metodo=metodo, lb=lb, vela=vela, señal=s,
            )

            if lot > 0.0:
                signal = Signal(direction=es_long, lot=lot)
                sid = self.gen_id(signal)
                positions[sid] = {'signal': signal, 'precio_in': precio_in}
                out.append((signal, sid))

        return out

    def _check_weekend_close(self) -> list[tuple[Signal, bytes]]:
        market = self._iterator.market
        last_updt = market.last_updt
        if last_updt is None:
            return []

        is_friday_close = (
            market.in_market
            and market.day == 4
            and market.minutes_left <= 1
        )
        is_sunday_forced_close = (
            market.weekends
            and market.day == 6
            and (self._positions_long or self._positions_short)
        )

        if not (is_friday_close or is_sunday_forced_close):
            return []

        ts = int(last_updt.timestamp())
        if self._last_weekend_close_ts == ts:
            return []
        self._last_weekend_close_ts = ts

        out: list[tuple[Signal, bytes]] = []
        for positions in (self._positions_long, self._positions_short):
            for sid, pos in list(positions.items()):
                pos['signal'].change_type()
                out.append((pos['signal'], sid))
                del positions[sid]

        return out

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
            self._opt_week(datetime.combine(monday, dtime.min, tzinfo=timezone.utc))

        else:
            self._params_long = {'met': vals['met_l'], 'vela': vals['vela_l'], 'lb': vals['lb_l']}
            self._params_short = {'met': vals['met_s'], 'vela': vals['vela_s'], 'lb': vals['lb_s']}

        self._minute_count = {True: 0, False: 0}

        return True

    def _load_week_cache(self, file_path) -> dict[date, dict]:
        if not file_path.exists():
            return {}
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

        print(monday_start, fryday_end)

        data_opt: pd.DataFrame = self._iterator.sub_data(monday_start, fryday_end)
        data_opt.rename(columns={'close': 'bid'}, inplace=True)
        data_opt["ask"] = data_opt["bid"] + data_opt["spread"]

        data_opt["Precio Spot"] = (data_opt["ask"] + data_opt["bid"]) / 2
        data_opt.drop(columns=['open', 'high', 'low', 'spread'], inplace=True)

        met_l, cand_l, lo_l = opti_main(data_opt, True, True, "fm", False)
        met_s, cand_s, lo_s = opti_main(data_opt, True, True, "fm", True)

        monday_key = monday_week.date()
        self._week_cache[monday_key] = {
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
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self._file_path, index=False)
