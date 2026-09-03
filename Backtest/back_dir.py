import os
import json
import pandas as pd
import numpy as np 
import random

from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any
import MetaTrader5 as mt5

import sys
sys.path.append(os.path.abspath('../optimal-moving-average'))

import keys
from find_best_dir import opti_dir
from find_best import opti_main

from use_tecnics import main
from read_data import ohlc_form

from direction_methods import DIR_METHODS, _split_signals_and_change
from tester import hit_ratio, rr_ratio, profit_ratio

from back_normal_reversion import (
    calcular_lotes_y_pnl, calcular_swap, obtener_comision_rt,
    _contabilizar, imprimir_tabla
)

keys.calls = 100

ASSET: str = "EURUSD_"
CAPITAL_LONG: float = 10_000
CAPITAL_SHORT: float = 10_000
APALANCAMIENTO: int = 2
# 2025-02-17
START_DATE: datetime = datetime(2024, 1, 1, 0, 0)
END_DATE: datetime = datetime(2024, 12, 31, 23, 59)

train_weeks: int = 5
test_time: int = 1

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


def parametros_random(shorts: bool) -> Dict[str, Any]:
    ma_method = random.choice(list(keys.methods))
    ma_candle = random.randint(keys.candles_min, keys.candles)
    ma_lookback = random.randint(keys.lookbacks_min, keys.lookbacks)

    dir_method = random.choice([
        "KEF", "HURST", "LO_MACKINLAY", "ADX",
        "VOL_RATIO", "SHANNON", "ATR_EXPANSION", "KELTNER_BREAKOUT",
    ])
    dir_candle = random.randint(1, keys.candles)
    dir_window = random.randint(2, 100)

    params = {
        "ma_method": ma_method,
        "ma_candle": ma_candle,
        "ma_lookback": ma_lookback,
        "name": dir_method,
        "candle": dir_candle,
        "window": dir_window,
    }

    if dir_method == "KEF":
        params["follow_tend"] = random.uniform(0.1, 1.0)
    elif dir_method == "HURST":
        params["follow_tend"] = random.uniform(0.5, 1.0)
    elif dir_method == "LO_MACKINLAY":
        params["k"] = random.randint(2, 10)
        params["follow_tend"] = random.uniform(0.5, 3.0)
    elif dir_method == "ADX":
        params["follow_tend"] = random.uniform(10.0, 40.0)
    elif dir_method == "VOL_RATIO":
        params["follow_tend"] = random.uniform(0.5, 3.0)
    elif dir_method == "SHANNON":
        params["bins"] = random.randint(5, 20)
        params["follow_tend"] = random.uniform(0.3, 1.0)
    elif dir_method == "ATR_EXPANSION":
        params["lookback_ma"] = random.randint(20, 100)
        params["follow_tend"] = random.uniform(1.0, 3.0)
    elif dir_method == "KELTNER_BREAKOUT":
        params["mult"] = random.uniform(1.0, 4.0)

    return params

def parametros_random_no_dir() -> Dict[str, Any]:
    ma_method = random.choice(list(keys.methods))
    ma_candle = random.randint(keys.candles_min, keys.candles)
    ma_lookback = random.randint(keys.lookbacks_min, keys.lookbacks)

    return {
        "ma_method": ma_method,
        "ma_candle": ma_candle,
        "ma_lookback": ma_lookback,
    }


def start_back_no_dir(usar_random: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    cache_suffix = "_NO_DIR_RANDOM" if usar_random else "_NO_DIR"
    params_cache = cargar_params_asset(ASSET + cache_suffix)

    lunes_test: datetime = adj_start
    side_key_l = "longs"
    side_key_s = "shorts"

    longs: list = []
    shorts: list = []

    while lunes_test < adj_end:
        lunes_key = lunes_test.strftime("%Y-%m-%d")
        viernes_end: datetime = (lunes_test + timedelta(weeks=test_time-1, days=4)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )

        if lunes_key not in params_cache:
            params_cache[lunes_key] = {}

        train_start = lunes_test - timedelta(weeks=train_weeks)
        train_end = (lunes_test - timedelta(days=3)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )

        cache_actualizada = False
        sub_data: pd.DataFrame = data.loc[train_start:viernes_end]

        if side_key_l in params_cache[lunes_key]:
            print(f"[{ASSET}{cache_suffix}] Carga local ({side_key_l}) para la semana: {lunes_key}")
            resultado_l = params_cache[lunes_key][side_key_l]["parametros"]
        else:
            if usar_random:
                print(f"[{ASSET}{cache_suffix}] Generando RANDOM ({side_key_l}) para la semana: {lunes_key}")
                parametros_random_l = parametros_random_no_dir()
                resultado_l = [
                    parametros_random_l["ma_method"],
                    parametros_random_l["ma_candle"],
                    parametros_random_l["ma_lookback"],
                ]
            else:
                print(f"[{ASSET}{cache_suffix}] Optimizando ({side_key_l}) para la semana: {lunes_key}")
                data_opt = data.loc[train_start:train_end]
                resultado_l = opti_main(data_opt, True, "fm", False)

            params_cache[lunes_key][side_key_l] = {"parametros": resultado_l}
            cache_actualizada = True

        ma_method_l, ma_candle_l, ma_lookback_l = resultado_l[0], resultado_l[1], resultado_l[2]

        data_real_calentada_l = ohlc_form(sub_data, ma_candle_l)

        signals_and_prices_l = main(
            ma_method_l,
            data_real_calentada_l[2]["close"],
            ma_lookback_l,
            ma_candle_l,
            sub_data,
            False,
            bid_df=data_real_calentada_l[0],
            ask_df=data_real_calentada_l[1],
        )

        longs.append(signals_and_prices_l.loc[lunes_test: viernes_end])

        if side_key_s in params_cache[lunes_key]:
            print(f"[{ASSET}{cache_suffix}] Carga local ({side_key_s}) para la semana: {lunes_key}")
            resultado_s = params_cache[lunes_key][side_key_s]["parametros"]
        else:
            if usar_random:
                print(f"[{ASSET}{cache_suffix}] Generando RANDOM ({side_key_s}) para la semana: {lunes_key}")
                parametros_random_s = parametros_random_no_dir()
                resultado_s = [
                    parametros_random_s["ma_method"],
                    parametros_random_s["ma_candle"],
                    parametros_random_s["ma_lookback"],
                ]
            else:
                print(f"[{ASSET}{cache_suffix}] Optimizando ({side_key_s}) para la semana: {lunes_key}")
                data_opt = data.loc[train_start:train_end]
                resultado_s = opti_main(data_opt, True, "fm", True)

            params_cache[lunes_key][side_key_s] = {"parametros": resultado_s}
            cache_actualizada = True

        ma_method_s, ma_candle_s, ma_lookback_s = resultado_s[0], resultado_s[1], resultado_s[2]

        data_real_calentada_s = ohlc_form(sub_data, ma_candle_s)

        signals_and_prices_s: pd.DataFrame = main(
            ma_method_s,
            data_real_calentada_s[2]["close"],
            ma_lookback_s,
            ma_candle_s, 
            sub_data,
            True,
            bid_df=data_real_calentada_s[0],
            ask_df=data_real_calentada_s[1],
        )

        shorts.append(signals_and_prices_s.loc[lunes_test: viernes_end])

        if cache_actualizada:
            guardar_params_asset(ASSET + cache_suffix, params_cache)

        lunes_test += timedelta(weeks=test_time)

    return pd.concat(longs), pd.concat(shorts)


def _safe_concat(lst: list) -> pd.DataFrame:
    return pd.concat(lst) if lst else pd.DataFrame()


def start_back(usar_random: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    cache_suffix = "_RANDOM" if usar_random else ""
    params_cache = cargar_params_asset(ASSET + cache_suffix)

    lunes_test: datetime = adj_start
    side_key_l = "longs"
    side_key_s = "shorts"

    longs_sin_cambio: list  = []
    shorts_cambiados: list  = []
    shorts_sin_cambio: list = []
    longs_cambiados: list   = []

    while lunes_test < adj_end:
        lunes_key = lunes_test.strftime("%Y-%m-%d")
        viernes_end: datetime = (lunes_test + timedelta(weeks=test_time-1, days=4)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )

        if lunes_key not in params_cache:
            params_cache[lunes_key] = {}

        train_start = lunes_test - timedelta(weeks=train_weeks)
        train_end = (lunes_test - timedelta(days=3)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )

        cache_actualizada = False
        sub_data: pd.DataFrame = data.loc[train_start:viernes_end]

        if side_key_l in params_cache[lunes_key]:
            print(f"[{ASSET}{cache_suffix}] Carga local ({side_key_l}) para la semana: {lunes_key}")
            parametros_l = params_cache[lunes_key][side_key_l]["parametros"]
            side_l = params_cache[lunes_key][side_key_l].get("side", "both")
        else:
            if usar_random:
                print(f"[{ASSET}{cache_suffix}] Generando RANDOM ({side_key_l}) para la semana: {lunes_key}")
                data_opt = data.loc[train_start:train_end]

                parametros_l = parametros_random(False)
                kpis_l = {}
                side_l = "both"
            else:
                print(f"[{ASSET}{cache_suffix}] Optimizando ({side_key_l}) para la semana: {lunes_key}")
                data_opt = data.loc[train_start:train_end]
                parametros_l, kpis_l = opti_dir(data_opt, True, False, keys.calls)
                side_l = kpis_l.get("filter", "both")

            params_cache[lunes_key][side_key_l] = {
                "parametros": parametros_l,
                "kpis": kpis_l,
                "side": side_l,
            }
            cache_actualizada = True

        data_real_calentada_l: pd.DataFrame = ohlc_form(sub_data, parametros_l["ma_candle"])

        signals_and_prices_l: pd.DataFrame = main(
            parametros_l["ma_method"],
            data_real_calentada_l[2]["close"],
            parametros_l["ma_lookback"],
            parametros_l["ma_candle"],
            sub_data,
            False,
            bid_df=data_real_calentada_l[0],
            ask_df=data_real_calentada_l[1],
        )

        changue_rules_l: pd.Series = DIR_METHODS[parametros_l["name"]](signals_and_prices_l, parametros_l, data_real_calentada_l[2])

        long, short = _split_signals_and_change(signals_and_prices_l, changue_rules_l, False, 
                                                sub_data, parametros_l["ma_candle"], data_real_calentada_l[0], data_real_calentada_l[1])

        if side_l in ("both", "revert"):
            longs_sin_cambio.append(long.loc[lunes_test: viernes_end])
        if side_l in ("both", "tend"):
            shorts_cambiados.append(short.loc[lunes_test: viernes_end])

        if side_key_s in params_cache[lunes_key]:
            print(f"[{ASSET}{cache_suffix}] Carga local ({side_key_s}) para la semana: {lunes_key}")
            parametros_s = params_cache[lunes_key][side_key_s]["parametros"]
            side_s = params_cache[lunes_key][side_key_s].get("side", "both")
        else:
            if usar_random:
                print(f"[{ASSET}{cache_suffix}] Generando RANDOM ({side_key_s}) para la semana: {lunes_key}")
                data_opt = data.loc[train_start:train_end]

                parametros_s = parametros_random(True)
                kpis_s = {}
                side_s = "both"
            else:
                print(f"[{ASSET}{cache_suffix}] Optimizando ({side_key_s}) para la semana: {lunes_key}")
                data_opt = data.loc[train_start:train_end]
                parametros_s, kpis_s = opti_dir(data_opt, True, True, keys.calls)
                side_s = kpis_s.get("filter", "both")

            params_cache[lunes_key][side_key_s] = {
                "parametros": parametros_s,
                "kpis": kpis_s,
                "side": side_s,
            }
            cache_actualizada = True

        data_real_calentada_s: pd.DataFrame = ohlc_form(sub_data, parametros_s["ma_candle"])

        signals_and_prices_s: pd.DataFrame = main(
            parametros_s["ma_method"],
            data_real_calentada_s[2]["close"],
            parametros_s["ma_lookback"],
            parametros_s["ma_candle"],
            sub_data,
            True,
            bid_df=data_real_calentada_s[0],
            ask_df=data_real_calentada_s[1],
        )

        changue_rules_s: pd.Series = DIR_METHODS[parametros_s["name"]](signals_and_prices_s, parametros_s, data_real_calentada_s[2])
        short, long = _split_signals_and_change(signals_and_prices_s, changue_rules_s, True, sub_data,
                                                parametros_s["ma_candle"], data_real_calentada_s[0], data_real_calentada_s[1])

        if side_s in ("both", "revert"):
            shorts_sin_cambio.append(short.loc[lunes_test: viernes_end])
        if side_s in ("both", "tend"):
            longs_cambiados.append(long.loc[lunes_test: viernes_end])

        if cache_actualizada:
            guardar_params_asset(ASSET + cache_suffix, params_cache)

        lunes_test += timedelta(weeks=test_time)

    return (
        _safe_concat(longs_sin_cambio),
        _safe_concat(shorts_cambiados),
        _safe_concat(shorts_sin_cambio),
        _safe_concat(longs_cambiados),
    )

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

def get_trades_diff(p: pd.DataFrame, short: bool) -> pd.Series:
    if p.empty:
        return pd.Series()
    pt = p["Prices"].diff()
    if short:
        return -pt[p["Signals"] == 1]

    return pt[p["Signals"] == -1]


def equity(p: pd.DataFrame, shorts: bool) -> float:
    if p.empty:
        return 0

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

    second_param = "shorts" if shorts else "longs"

    capital_inicial = CAPITAL_SHORT if shorts else CAPITAL_LONG
    mon = capital_inicial
    open_sig = -1 if shorts else 1
    cantidad = 0.0

    multiplicador = 1.0
    riesgo = .01

    while lunes_test < adj_end:
        lunes_key = lunes_test.strftime("%Y-%m-%d")
        viernes_end: datetime = (lunes_test + timedelta(weeks=test_time-1, days=4)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )

        params = params_cache[lunes_key][second_param]["parametros"]
        sub_data = data[lunes_test: viernes_end]
        signals_and_prices = p[lunes_test: viernes_end]

        candle_ma = params["ma_candle"]
        ohlc = ohlc_form(sub_data, candle_ma)

        if shorts:
            ohlc = ohlc[1]
        else:
            ohlc = ohlc[0]

        entrys = signals_and_prices[signals_and_prices["Signals"] == open_sig]
        exits = signals_and_prices[signals_and_prices["Signals"] == -open_sig]

        for k in range(len(entrys)):
            indice_entrada = entrys.index[k]
            indice_salida  = exits.index[k]

            precio_entrada = signals_and_prices["Prices"][indice_entrada]
            cantidad = (capital_inicial * APALANCAMIENTO * multiplicador)/precio_entrada

            precio_salida = signals_and_prices["Prices"][indice_salida]

            sub_ohlc = ohlc[(indice_entrada < ohlc.index) & (ohlc.index <= indice_salida)]

            stop_loss_price = capital_inicial * riesgo

            if shorts:
                pnl_latente = (precio_entrada - sub_ohlc["high"]) * cantidad

            else:
                pnl_latente = (sub_ohlc["low"] - precio_entrada) * cantidad

            sl_tocados = pnl_latente[pnl_latente <= -stop_loss_price]

            if not sl_tocados.empty:
                pnl = -stop_loss_price
            else:
                pnl = (precio_salida - precio_entrada) * cantidad
                if shorts:
                    pnl = -pnl

            mon += pnl

            if pnl < 0 and multiplicador < 8:
                multiplicador *= 2

            else:
                multiplicador = 1

        lunes_test += timedelta(weeks=test_time)

    return mon - capital_inicial

if __name__ == "__main__":
    p, q, r, s = start_back()
    print(p)
    print(equity(p, False), equity(q, True), equity(r, True), equity(s, False))

    p = get_trades_diff(p, False)
    q = get_trades_diff(q, True)
    r = get_trades_diff(r, True)
    s = get_trades_diff(s, False)

    tr = pd.concat([p, q, r, s])

    print(hit_ratio(p), rr_ratio(tr), profit_ratio(p), len(p))
    print(hit_ratio(q), rr_ratio(q), profit_ratio(q), len(q))
    print(hit_ratio(r), rr_ratio(r), profit_ratio(r), len(r))
    print(hit_ratio(s), rr_ratio(s), profit_ratio(s), len(s))
