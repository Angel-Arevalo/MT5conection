from datetime import datetime, timezone
from time import sleep
from typing import Optional, Tuple
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

class DataIterator:
    name_asset: str
    _backtest: bool
    
    _point_asset: float
    _bid_data: Optional[pd.DataFrame]
    
    _current_index: int
    _total_candles: int

    def __init__(self, name_asset: str) -> None:
        if not mt5.initialize():
            print(f"Error al inicializar MT5: {mt5.last_error()}")
            return

        self.name_asset = name_asset
        self._backtest = False
        self._point_asset = 0.0
        self._bid_data = None
        self._current_index = 0
        self._total_candles = 0

        symbol_info: Optional[mt5.SymbolInfo] = mt5.symbol_info(self.name_asset)
        if symbol_info is not None:
            self._point_asset = float(symbol_info.point)
        else:
            print(f"Símbolo {self.name_asset} no encontrado.")
            return

    def backtest(self, start_back: datetime, end_back: datetime) -> None:
        self._backtest = True
        self._get_mt5_info(start_back, end_back)

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

        self._bid_data = data_cruda[['open', 'high', 'low', 'close', 'spread']].copy()
        self._current_index = 0
        self._total_candles = len(data_cruda.index)

    def next_candle(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._backtest:
            if self._bid_data is None or self._current_index >= self._total_candles:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

            row_array: np.ndarray = self._bid_data.iloc[self._current_index].to_numpy()
            self._current_index += 1

            bid_array: np.ndarray = row_array[:4]
            spread_value: float = float(row_array[4])
            ask_array: np.ndarray = bid_array + (spread_value * self._point_asset)

            return bid_array, ask_array

        else:
            now: datetime = datetime.now(timezone.utc)
            seconds_to_sleep: float = 60.0 - (now.second + now.microsecond / 1_000_000.0) + 0.5
            sleep(seconds_to_sleep)

            rates: Optional[np.ndarray] = mt5.copy_rates_from_pos(
                self.name_asset, 
                mt5.TIMEFRAME_M1, 
                1, 
                1
            )

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
            ask_array: np.ndarray = bid_array + (spread_value * self._point_asset)

            return bid_array, ask_array

     def sub_data(self, start_time: datetime, end_time: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sub_df: pd.DataFrame

        if self._backtest:
            if self._bid_data is None:
                return pd.DataFrame(), pd.DataFrame()

            if end_time is None:
                end_time = self._bid_data.index[-1]

            if end_time <= start_time: 
                raise ValueError("Periodo no válido")

            if end_time > self._bid_data.index[-1] or end_time > self.bid_data.index[self._current_index]:
                raise ValueError("Datos no vistos")

            sub_df = self._bid_data.loc[start_time:end_time]
            
        else:
            if end_time is None:
                end_time = datetime.now()

            rates: Optional[np.ndarray] = mt5.copy_rates_range(
                self.name_asset,
                mt5.TIMEFRAME_M1,
                start_time,
                end_time
            )

            if rates is None or len(rates) == 0:
                print(f"No se pudieron obtener datos en vivo para {self.name_asset}.")
                return pd.DataFrame(), pd.DataFrame()

            data_cruda: pd.DataFrame = pd.DataFrame(rates)
            data_cruda.index = pd.to_datetime(data_cruda["time"], unit="s")
            sub_df = data_cruda

        sub_bid: pd.DataFrame = sub_df[['open', 'high', 'low', 'close']].copy()
        spread_series: pd.Series = sub_df['spread']
        sub_ask: pd.DataFrame = sub_bid.add(spread_series * self._point_asset, axis=0)

        return sub_bid, sub_ask
