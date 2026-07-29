from datetime import datetime, timezone, timedelta, time as dtime
from time import sleep

from typing import Optional, Tuple
import MetaTrader5 as mt5

import numpy as np
import pandas as pd

import secrets
from MarketStatus import MarketStatus

class DataIterator:
    name_asset: str
    _backtest: bool

    _point_asset: float
    _bid_data: Optional[pd.DataFrame]

    _current_index: int
    _total_candles: int

    _market: MarketStatus
    __market_key: bytes

    def __init__(self, name_asset: str) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"Error al inicializar MT5: {mt5.last_error()}")

        self.name_asset = name_asset
        self._backtest = False

        self._point_asset = 0.0
        self._bid_data = None

        self._current_index = 0
        self._total_candles = 0

        self._last_date = 0
        self.__market_key: bytes = secrets.token_bytes(16)

        symbol_info: Optional[mt5.SymbolInfo] = mt5.symbol_info(self.name_asset)
        if symbol_info is None:
            mt5.shutdown()
            raise ValueError(f"Símbolo {self.name_asset} no encontrado en MT5.")

        self._point_asset = float(symbol_info.point)

        self._get_bursatil_interval()

    def backtest(self, start_back: datetime, end_back: Optional[datetime] = None) -> None:
        self._backtest = True

        if end_back is None:
            end_back = start_back

        start_time = start_back.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = end_back.replace(hour=23, minute=59, second=59, microsecond=0)

        self._get_mt5_info(start_time, end_time)

    def _get_mt5_info(self, start_time: datetime, end_time: datetime) -> None:
        rates: Optional[np.ndarray] = mt5.copy_rates_range(
            self.name_asset,
            mt5.TIMEFRAME_M1,
            start_time,
            end_time
        )

        if rates is None or len(rates) == 0:
            print(f"No se encontraron datos para {self.name_asset}.")
            return

        data_cruda: pd.DataFrame = pd.DataFrame(rates)
        data_cruda.index = pd.to_datetime(data_cruda["time"], unit="s")

        self._bid_data = data_cruda[['time', 'open', 'high', 'low', 'close', 'spread']].copy()
        print(self._bid_data)
        self._current_index = 0
        self._total_candles = len(data_cruda.index)

    def _get_bursatil_interval(self) -> None:
        hoy = datetime.now(timezone.utc)
        wednesday_this_week = hoy - timedelta(days=hoy.weekday() - 2)

        wednesday = wednesday_this_week - timedelta(weeks=1)
        saturday = hoy - timedelta(days=hoy.weekday() - 5) - timedelta(weeks=1)

        rates: Optional[np.ndarray] = mt5.copy_rates_range(
            self.name_asset, mt5.TIMEFRAME_M1,
            wednesday.replace(hour=0, minute=0, second=0, microsecond=0),
            wednesday.replace(hour=23, minute=59, second=59, microsecond=0)
        )
 
        if rates is not None and len(rates) > 0:
            start_day = datetime.fromtimestamp(int(rates[0]['time']), tz=timezone.utc).time()
            end_day = datetime.fromtimestamp(int(rates[-1]['time']), tz=timezone.utc).time()

        saturday_rates = mt5.copy_rates_range(
            self.name_asset, mt5.TIMEFRAME_M1,
            saturday.replace(hour=0, minute=0, second=0, microsecond=0),
            saturday.replace(hour=23, minute=59, second=59, microsecond=0)
        )

        trades_weekends = saturday_rates is not None and len(saturday_rates) > 0
        self._market = MarketStatus(start_day, end_day, trades_weekends, self.__market_key)

    def next_candle(self) -> Tuple[np.ndarray, float]:
        if self._backtest:
            if self._bid_data is None or self._current_index >= self._total_candles:
                self._backtest = False
                return np.array([], dtype=np.float64), 0.0

            row = self._bid_data.iloc[self._current_index]
            self._current_index += 1

            self._last_date = int(row['time'])

            bid_array: np.ndarray = row[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float64)
            spread_value: float = float(row['spread'])

        else:
            now: datetime = datetime.now(timezone.utc)

            if self._last_date == 0:
                self._last_date = mt5.copy_rates_from_pos(self.name_asset, 
                    mt5.TIMEFRAME_M1, 
                    1,
                    1
                )[0][0]

            seconds_to_sleep: float = 60.0 - (now.second)

            sleep(seconds_to_sleep)

            new_candle = False

            while not new_candle:
                rates: Optional[np.ndarray] = mt5.copy_rates_from_pos(
                    self.name_asset,
                    mt5.TIMEFRAME_M1,
                    1,
                    1
                )

                if rates[0][0] == self._last_date:
                    sleep(1)
                else:
                    new_candle = True
                    self._last_date = rates[0][0]

            if rates is None or len(rates) == 0:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

            closed_candle: np.void = rates[0]

            bid_array: np.ndarray = np.array([
                float(closed_candle['open']),
                float(closed_candle['high']),
                float(closed_candle['low']),
                float(closed_candle['close'])
            ], dtype=np.float64)

            spread_value: float = float(closed_candle['spread'])

        self._market.update(self._last_date, self.__market_key)

        return bid_array, spread_value * self._point_asset

    @property
    def market(self) -> MarketStatus:
        return self._market

    def sub_data(self, start_time: datetime, end_time: Optional[datetime] = None) -> pd.DataFrame:
        if self._backtest:
            if self._bid_data is None or self._current_index == 0:
                print("No hay datos cargados o el backtest no ha iniciado.")
                return pd.DataFrame()

            max_allowed_time = pd.to_datetime(self._last_date, unit="s", utc=True)

            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)

            if end_time is None or end_time > max_allowed_time:
                end_time = max_allowed_time
            elif end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)

            sub_df = self._bid_data.loc[start_time:end_time].copy()

        else:
            now_utc = datetime.now(timezone.utc)
            if end_time is None or end_time > now_utc:
                end_time = now_utc

            rates: Optional[np.ndarray] = mt5.copy_rates_range(
                self.name_asset,
                mt5.TIMEFRAME_M1,
                start_time,
                end_time
            )

            if rates is None or len(rates) == 0:
                print(f"No se pudieron obtener datos en vivo para {self.name_asset}.")
                return pd.DataFrame()

            data_cruda: pd.DataFrame = pd.DataFrame(rates)
            data_cruda.index = pd.to_datetime(data_cruda["time"], unit="s", utc=True)
            sub_df = data_cruda[['open', 'high', 'low', 'close', 'spread']].copy()

        if sub_df.empty:
            return pd.DataFrame()

        result_df = sub_df[['open', 'high', 'low', 'close', 'spread']].copy()
        result_df['spread'] = result_df['spread'] * self._point_asset

        return result_df
