from datetime import datetime, timezone
from time import sleep
from typing import Optional, Tuple
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

class DataIterator:
    name_asset: str
    _backtest: bool

    _start_date: datetime
    _end_date: datetime

    _point_asset: float
    _bid_data: Optional[pd.DataFrame]

    _current_index: int
    _total_candles: int

    def __init__(self, name_asset: str, backtest: bool, start_back: datetime, end_back: datetime) -> None:
        if not mt5.initialize():
            print(f"Error al inicializar MT5: {mt5.last_error()}")
            return

        self.name_asset = name_asset
        self._backtest = backtest
        self._start_date = start_back
        self._end_date = end_back

        self._bid_data = None
        self._current_index = 0
        self._total_candles = 0
        self._point_asset = 0.0

        symbol_info: Optional[mt5.SymbolInfo] = mt5.symbol_info(self.name_asset)
        if symbol_info is not None:
            self._point_asset = float(symbol_info.point)
        else:
            print(f"Símbolo {self.name_asset} no encontrado.")
            return

        if self._backtest:
            self._get_mt5_info()

    def _get_mt5_info(self) -> None:
        rates: Optional[np.ndarray] = mt5.copy_rates_range(
            self.name_asset,
            mt5.TIMEFRAME_M1,
            self._start_date,
            self._end_date
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
            if self._bid_data is None:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

            if self._current_index >= self._total_candles:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
 
            self._current_index += 1

            row_array: np.ndarray = self._bid_data.iloc[self._current_index - 1].to_numpy()
            bid_array: np.ndarray = row_array[:4]
            spread_value: float = float(row_array[4])
            ask_array: np.ndarray = bid_array + (spread_value * self._point_asset)

            return bid_array, ask_array

        else:
            seconds_to_sleep: float = (60.0 - datetime.now(timezone.utc).second) + 0.5
            sleep(seconds_to_sleep)

            rates: Optional[np.ndarray] = mt5.copy_rates_from_pos(self.name_asset, mt5.TIMEFRAME_M1, 1, 1)

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
