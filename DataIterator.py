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

    _times: Optional[np.ndarray]
    _ohlc: Optional[np.ndarray]
    _spreads: Optional[np.ndarray]

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

        self._times = None
        self._ohlc = None
        self._spreads = None

        self._last_date = 0
        self.__market_key: bytes = secrets.token_bytes(16)

        symbol_info: Optional[mt5.SymbolInfo] = mt5.symbol_info(self.name_asset)
        if symbol_info is None:
            mt5.shutdown()
            raise ValueError(f"Símbolo {self.name_asset} no encontrado en MT5.")

        self._point_asset = float(symbol_info.point)

        self._get_bursatil_interval()
        self._update_market()

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
        data_cruda.index = pd.to_datetime(data_cruda["time"], unit="s", utc=True)

        self._bid_data = data_cruda[['time', 'open', 'high', 'low', 'close', 'spread']].copy()
        self._current_index = 0
        self._total_candles = len(data_cruda.index)

        self._times = self._bid_data['time'].to_numpy(dtype=np.int64)
        self._ohlc = self._bid_data[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float64)
        self._spreads = self._bid_data['spread'].to_numpy(dtype=np.float64)

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
 
        start_day = dtime(0, 0)
        end_day = dtime(23, 59)

        if rates is not None and len(rates) > 0:
            start_day = datetime.fromtimestamp(int(rates[0]['time']), tz=timezone.utc).time()
            end_day = (datetime.fromtimestamp(int(rates[-1]['time']), tz=timezone.utc) - timedelta(minutes = 1)).time()

        saturday_rates = mt5.copy_rates_range(
            self.name_asset, mt5.TIMEFRAME_M1,
            saturday.replace(hour=0, minute=0, second=0, microsecond=0),
            saturday.replace(hour=23, minute=59, second=59, microsecond=0)
        )

        trades_weekends = saturday_rates is not None and len(saturday_rates) > 0
        self._market = MarketStatus(start_day, end_day, trades_weekends, self.__market_key)

    def _update_market(self) -> None:
        rates = mt5.copy_rates_from_pos(self.name_asset, mt5.TIMEFRAME_M1, 1, 1)

        if rates is not None and len(rates) > 0:
            self._market.update(int(rates[0][0]), self.__market_key)

    def __wait_swap_market(self) -> None:
        while True:
            rates_init = mt5.copy_rates_from_pos(self.name_asset, mt5.TIMEFRAME_M1, 0, 1)
            if rates_init is not None and len(rates_init) > 0:
                if datetime.fromtimestamp(rates_init[0][0], tz=timezone.utc).time() == self._market.start_day:
                    return

            now: datetime = datetime.now(timezone.utc)
            sleep(60 - now.second + 10)

    def next_candle(self) -> Tuple[np.ndarray, float]:
        if self._backtest:
            if self._ohlc is None or self._current_index >= self._total_candles:
                self._backtest = False
                self._last_date = 0
                self._update_market()
                return np.array([], dtype=np.float64), 0.0

            idx = self._current_index
            self._current_index += 1

            self._last_date = int(self._times[idx])

            bid_array: np.ndarray = self._ohlc[idx]
            spread_value: float = float(self._spreads[idx])

        else:
            if self._last_date == 0:
                rates_init = mt5.copy_rates_from_pos(self.name_asset, mt5.TIMEFRAME_M1, 1, 1)
                if rates_init is not None and len(rates_init) > 0:
                    self._last_date = int(rates_init[0][0])


            if datetime.fromtimestamp(self._last_date, tz=timezone.utc).time() == self._market.end_day:
                self.__wait_swap_market()

            now: datetime = datetime.now(timezone.utc)

            seconds_to_sleep: float = 60.0 - now.second
            sleep(seconds_to_sleep)

            new_candle = False
            rates = None

            while not new_candle:
                rates = mt5.copy_rates_from_pos(
                    self.name_asset,
                    mt5.TIMEFRAME_M1,
                    1,
                    1
                )

                if rates is None or len(rates) == 0:
                    sleep(1)
                    continue

                current_candle_time = int(rates[0][0])

                if current_candle_time == self._last_date:
                    sleep(1)
                    continue

                else:
                    new_candle = True
                    self._last_date = current_candle_time

            closed_candle: np.void = rates[0]

            bid_array = np.array([
                float(closed_candle['open']),
                float(closed_candle['high']),
                float(closed_candle['low']),
                float(closed_candle['close'])
            ], dtype=np.float64)

            spread_value = float(closed_candle['spread'])

        self._market.update(self._last_date, self.__market_key)

        return bid_array, spread_value * self._point_asset


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

            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())

            i0 = int(np.searchsorted(self._times, start_ts, side="left"))
            i1 = int(np.searchsorted(self._times, end_ts, side="right"))

            sub_df = pd.DataFrame(
                self._ohlc[i0:i1],
                columns=['open', 'high', 'low', 'close'],
                index=pd.to_datetime(self._times[i0:i1], unit="s", utc=True)
            )
            sub_df['spread'] = self._spreads[i0:i1]

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

    @property
    def market(self) -> MarketStatus:
        return self._market

    @property
    def iterator_type(self) -> bool:
        return self._backtest
