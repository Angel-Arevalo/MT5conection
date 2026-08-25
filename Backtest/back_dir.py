import os
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any
import MetaTrader5 as mt5

import sys
sys.path.append(os.path.abspath('../optimal-moving-average'))

import keys
from find_best_dir import opti_dir

from use_tecnics import main
from read_data import ohlc_form

keys.calls = 20

ASSET: str = "EURUSD_"
CAPITAL_LONG: float = 10_000
CAPITAL_SHORT: float = 10_000
APALANCAMIENTO: int = 2

START_DATE: datetime = datetime(2024, 1, 8, 0, 0)
END_DATE: datetime = datetime(2024, 12, 31, 23, 59)

train_weeks: int = 16

PARAMS_DIR = "dir_params"

def obtener_ruta_json(asset: str) -> str:
    if not os.path.exists(PARAMS_DIR):
        os.makedirs(PARAMS_DIR)

    clean_asset_name = asset.replace("/", "").replace("\\", "")
    return os.path.join(PARAMS_DIR, f"{clean_asset_name}.json")

def cargar_params_asset(asset: str) -> Dict[str, Any]:
    filepath = obtener_ruta_json(asset)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def guardar_params_asset(asset: str, data: Dict[str, Any]) -> None:
    filepath = obtener_ruta_json(asset)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def start_back(shorts: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adj_start = START_DATE
    adj_end = END_DATE

    dias_para_lunes: int = (0 - adj_start.weekday()) % 7
    if dias_para_lunes != 0:
        adj_start = (adj_start + timedelta(days=dias_para_lunes)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    dias_para_viernes: int = (adj_end.weekday() - 4) % 7
    if dias_para_viernes != 0:
        adj_end = (adj_end - timedelta(days=dias_para_viernes)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )

    if adj_start > adj_end:
        raise ValueError("La fecha de inicio ajustada es posterior a la fecha de fin.")

    data = pedir_data_mt5(ASSET, adj_start, adj_end)

    params_cache = cargar_params_asset(ASSET)

    lunes_test: datetime = adj_start
    side_key = "shorts" if shorts else "longs"

    while lunes_test < adj_end:
        lunes_key = lunes_test.strftime("%Y-%m-%d")

        if lunes_key in params_cache and side_key in params_cache[lunes_key]:
            print(f"[{ASSET}] Carga local ({side_key}) para la semana: {lunes_key}")
            parametros = params_cache[lunes_key][side_key]["parametros"]
            kpis = params_cache[lunes_key][side_key]["kpis"]

        else:
            print(f"[{ASSET}] Optimizando ({side_key}) para la semana: {lunes_key}")
            train_start = lunes_test - timedelta(weeks=train_weeks)
            train_end = (lunes_test - timedelta(days=3)).replace(
                hour=23, minute=59, second=59, microsecond=0
            )

            data_opt = data.loc[train_start:train_end]

            parametros, kpis = opti_dir(data_opt, True, shorts, keys.calls)

            if lunes_key not in params_cache:
                params_cache[lunes_key] = {}

            params_cache[lunes_key][side_key] = {
                "parametros": parametros,
                "kpis": kpis
            }

            guardar_params_asset(ASSET, params_cache)

        viernes_end: datetime = (lunes_test + timedelta(days=4)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )

        data_real_calentada: pd.DataFrame = ohlc_form(data.loc[train_start: viernes_end], parametros["ma_candle"])
        signals_and_prices: pd.DataFrame = main(parametros["ma_method"], data_real_calentada, parametros["ma_lookback"], data.loc[train_start: viernes_end])

        lunes_test += timedelta(weeks=1)

    return None, None

def pedir_data_mt5(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError(f"Error al inicializar MT5: {mt5.last_error()}")

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        mt5.shutdown()
        raise ValueError(f"No se encontró el símbolo {symbol}")

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            mt5.shutdown()
            raise RuntimeError(f"No se pudo activar el símbolo {symbol}")

    point: float = symbol_info.point

    fetch_start = start - timedelta(weeks=train_weeks)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, fetch_start, end)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError("No se obtuvieron datos de MT5 para el rango especificado.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"close": "bid"})
    df["ask"] = df["bid"] + (df["spread"] * point)

    return df[["time", "bid", "ask"]].set_index("time")


if __name__ == "__main__":
    start_back(True)
