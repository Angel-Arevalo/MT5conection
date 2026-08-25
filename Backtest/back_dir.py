import os
import json
import pandas as pd
import numpy as np 

from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any
import MetaTrader5 as mt5

import sys
sys.path.append(os.path.abspath('../optimal-moving-average'))

import keys
from find_best_dir import opti_dir

from use_tecnics import main
from read_data import ohlc_form

from direction_methods import DIR_METHODS, _split_signals_and_change
from tester import hit_ratio, rr_ratio, profit_ratio

from back_normal_reversion import (
    calcular_lotes_y_pnl, calcular_swap, obtener_comision_rt,
    _contabilizar, imprimir_tabla
)

keys.calls = 20

ASSET: str = "EURUSD_"
CAPITAL_LONG: float = 10_000
CAPITAL_SHORT: float = 10_000
APALANCAMIENTO: int = 2

START_DATE: datetime = datetime(2024, 1, 8, 0, 0)
END_DATE: datetime = datetime(2026, 8, 21, 23, 59)

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

def start_back() -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    side_key_l = "longs"
    side_key_s = "shorts"

    longs_sin_cambio: list  = []
    shorts_cambiados: list  = []
    shorts_sin_cambio: list = []
    longs_cambiados: list   = []

    while lunes_test < adj_end:
        lunes_key = lunes_test.strftime("%Y-%m-%d")
        viernes_end: datetime = (lunes_test + timedelta(days=4)).replace(
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
            print(f"[{ASSET}] Carga local ({side_key_l}) para la semana: {lunes_key}")
            parametros_l = params_cache[lunes_key][side_key_l]["parametros"]
        else:
            print(f"[{ASSET}] Optimizando ({side_key_l}) para la semana: {lunes_key}")
            data_opt = data.loc[train_start:train_end]
            parametros_l, kpis_l = opti_dir(data_opt, True, False, keys.calls)

            params_cache[lunes_key][side_key_l] = {
                "parametros": parametros_l,
                "kpis": kpis_l
            }
            cache_actualizada = True

        data_real_calentada_l: pd.DataFrame = ohlc_form(sub_data, parametros_l["ma_candle"])

        signals_and_prices_l: pd.DataFrame = main(
            parametros_l["ma_method"], 
            data_real_calentada_l[2]["close"], 
            parametros_l["ma_lookback"], 
            False,
            sub_data
        )

        changue_rules_l: pd.Series = DIR_METHODS[parametros_l["name"]](signals_and_prices_l, parametros_l, data_real_calentada_l[2])
        long, short = _split_signals_and_change(signals_and_prices_l, changue_rules_l, False, sub_data)

        longs_sin_cambio.append(long.loc[lunes_test: viernes_end])
        shorts_cambiados.append(short.loc[lunes_test: viernes_end])

        if side_key_s in params_cache[lunes_key]:
            print(f"[{ASSET}] Carga local ({side_key_s}) para la semana: {lunes_key}")
            parametros_s = params_cache[lunes_key][side_key_s]["parametros"]
        else:
            print(f"[{ASSET}] Optimizando ({side_key_s}) para la semana: {lunes_key}")
            data_opt = data.loc[train_start:train_end]
            parametros_s, kpis_s = opti_dir(data_opt, True, True, keys.calls)

            params_cache[lunes_key][side_key_s] = {
                "parametros": parametros_s,
                "kpis": kpis_s
            }
            cache_actualizada = True

        data_real_calentada_s: pd.DataFrame = ohlc_form(sub_data, parametros_s["ma_candle"])

        signals_and_prices_s: pd.DataFrame = main(
            parametros_s["ma_method"], 
            data_real_calentada_s[2]["close"], 
            parametros_s["ma_lookback"], 
            True,
            sub_data
        )

        changue_rules_s: pd.Series = DIR_METHODS[parametros_s["name"]](signals_and_prices_s, parametros_s, data_real_calentada_s[2])
        short, long = _split_signals_and_change(signals_and_prices_s, changue_rules_s, True, sub_data)

        shorts_sin_cambio.append(short.loc[lunes_test: viernes_end])
        longs_cambiados.append(long.loc[lunes_test: viernes_end])

        if cache_actualizada:
            guardar_params_asset(ASSET, params_cache)

        lunes_test += timedelta(weeks=1)

    return pd.concat(longs_sin_cambio), pd.concat(shorts_cambiados), pd.concat(shorts_sin_cambio), pd.concat(longs_cambiados)

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
    pt = p["Prices"].diff()
    if short:
        return -pt[p["Signals"] == 1]

    return pt[p["Signals"] == -1]

def equity(p: pd.DataFrame, short: bool) -> pd.Series:
    capital_inicial = CAPITAL_SHORT if short else CAPITAL_LONG
    mon = capital_inicial
    open_sig = -1 if short else 1
    cantidad = 0.0
    precio_entrada = 0.0

    for signal, precio in zip(p["Signals"], p["Prices"]):
        if signal == open_sig:
            cantidad = (capital_inicial * APALANCAMIENTO) / precio
            precio_entrada = precio
        else:
            pnl = (precio - precio_entrada) * cantidad
            if short:
                pnl = -pnl
            mon += pnl

    return mon - capital_inicial



def contabilizar_realista(trades, capital_inicial, swap_rate, swap_mode,
                           contract_size, tick_value, tick_size, rollover3days, comision_rt):
    df = pd.DataFrame(trades)
    if df.empty:
        return df, capital_inicial

    bal = capital_inicial
    pnl_bruto_l = []; pnl_neto_l = []; bal_l = []
    lot_l = []; swp_l = []; com_l = []; ret_l = []

    for _, row in df.iterrows():
        lotes, pnl_bruto = calcular_lotes_y_pnl(
            row['precio_entrada'], row['pnl_price'], bal, contract_size)
        com = comision_rt * lotes
        swap = calcular_swap(
            row['fecha_entrada'], row['fecha_salida'],
            lotes, swap_rate, swap_mode,
            row['precio_entrada'], contract_size,
            tick_value, tick_size, rollover3days)

        pnl_neto = pnl_bruto - com + swap

        prev = bal
        bal += pnl_neto  # balance real, sin apartar nada

        pnl_bruto_l.append(pnl_bruto); pnl_neto_l.append(pnl_neto)
        bal_l.append(bal); lot_l.append(lotes)
        swp_l.append(swap); com_l.append(com)
        ret_l.append(pnl_neto / prev if prev != 0 else 0.0)

    df['lotes']         = lot_l
    df['swap_usd']       = swp_l
    df['comision_usd']   = com_l
    df['pnl_bruto_usd']  = pnl_bruto_l
    df['pnl_neto_usd']   = pnl_neto_l
    df['retorno_trade']  = ret_l
    df['balance_real']   = bal_l

    df['peak']         = np.maximum(df['balance_real'].cummax(), capital_inicial)
    df['drawdown_abs'] = df['peak'] - df['balance_real']
    df['drawdown_pct'] = df['drawdown_abs'] / df['peak'] * 100

    return df, bal


def imprimir_tabla_realista(lista_dfs, lado: str, capital_inicial: float):
    dfs_validos = [d for d in lista_dfs if isinstance(d, pd.DataFrame) and not d.empty]
    if not dfs_validos:
        print(f"\n[{lado}] Sin operaciones registradas.")
        return

    df = pd.concat(dfs_validos, ignore_index=True)

    total_trades = len(df)
    wins         = df[df['pnl_neto_usd'] > 0]
    losses       = df[df['pnl_neto_usd'] <= 0]
    win_rate     = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    pnl_bruto  = df['pnl_bruto_usd'].sum()
    comisiones = df['comision_usd'].sum()
    swaps      = df['swap_usd'].sum()
    pnl_neto   = df['pnl_neto_usd'].sum()

    gross_profit  = wins['pnl_neto_usd'].sum()
    gross_loss    = abs(losses['pnl_neto_usd'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    max_dd_pct    = df['drawdown_pct'].max() if 'drawdown_pct' in df.columns else 0.0

    capital_final = capital_inicial + pnl_neto

    print(f"\n{'='*65}")
    print(f" RESUMEN REALISTA — {lado}")
    print(f"{'='*65}")
    print(f"  Total Trades        : {total_trades}")
    print(f"  Trades Ganadores    : {len(wins)} ({win_rate:.2f}%)")
    print(f"  Trades Perdedores   : {len(losses)}")
    print(f"  -------------------------------------------")
    print(f"  Capital Inicial     : ${capital_inicial:,.2f}")
    print(f"  PnL Bruto (USD)     : ${pnl_bruto:,.2f}")
    print(f"  Comisiones (USD)    : ${comisiones:,.2f}")
    print(f"  Swaps (USD)         : ${swaps:,.2f}")
    print(f"  -------------------------------------------")
    print(f"  Capital FINAL real  : ${capital_final:,.2f}")
    print(f"  Retorno total       : {(pnl_neto / capital_inicial * 100):.2f}%")
    print(f"  Profit Factor       : {profit_factor:.2f}")
    print(f"  Max Drawdown (%)    : {max_dd_pct:.2f}%")
    print(f"{'='*65}\n")


def evaluar_realista(p: pd.DataFrame, q: pd.DataFrame, r: pd.DataFrame, s: pd.DataFrame,
                      symbol: str = ASSET) -> tuple:
    if not mt5.initialize():
        raise RuntimeError(f"Error al inicializar MT5: {mt5.last_error()}")
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"No se pudo seleccionar {symbol}")

    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise ValueError(f"No se encontró el símbolo {symbol}")

    contract_size = info.trade_contract_size
    tick_size     = info.trade_tick_size
    tick_value    = info.trade_tick_value
    swap_mode     = info.swap_mode
    swap_long_r   = info.swap_long
    swap_short_r  = info.swap_short
    rollover3days = info.swap_rollover3days

    comision_rt = obtener_comision_rt(symbol, datetime.now().year)
    mt5.shutdown()

    trades_long  = trades_df_to_records(p, False) + trades_df_to_records(s, False)
    trades_short = trades_df_to_records(q, True)  + trades_df_to_records(r, True)

    df_long, _  = contabilizar_realista(
        trades_long, CAPITAL_LONG, swap_long_r, swap_mode,
        contract_size, tick_value, tick_size, rollover3days, comision_rt)
    df_short, _ = contabilizar_realista(
        trades_short, CAPITAL_SHORT, swap_short_r, swap_mode,
        contract_size, tick_value, tick_size, rollover3days, comision_rt)

    imprimir_tabla_realista([df_long],  "LONG",  CAPITAL_LONG)
    imprimir_tabla_realista([df_short], "SHORT", CAPITAL_SHORT)

    return df_long, df_short

def trades_df_to_records(p: pd.DataFrame, short: bool) -> list:
    open_sig = -1 if short else 1
    idx = p.index.to_numpy()
    signals = p["Signals"].to_numpy()
    prices = p["Prices"].to_numpy()

    records = []
    for i in range(0, len(p) - 1, 2):
        if signals[i] != open_sig:
            continue

        precio_entrada = float(prices[i])
        precio_salida = float(prices[i + 1])
        pnl_price = (precio_salida - precio_entrada) if not short else (precio_entrada - precio_salida)

        records.append({
            "fecha_entrada": pd.Timestamp(idx[i]),
            "fecha_salida": pd.Timestamp(idx[i + 1]),
            "precio_entrada": precio_entrada,
            "precio_salida": precio_salida,
            "pnl_price": pnl_price,
            "tipo": "Cierre_Señal",
            "mae_price": 0.0,
            "mfe_price": 0.0,
        })
    return records


def evaluar_con_costos(p: pd.DataFrame, q: pd.DataFrame, r: pd.DataFrame, s: pd.DataFrame,
                        symbol: str = ASSET) -> tuple:
    if not mt5.initialize():
        raise RuntimeError(f"Error al inicializar MT5: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"No se pudo seleccionar {symbol}")

    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise ValueError(f"No se encontró el símbolo {symbol}")

    contract_size = info.trade_contract_size
    tick_size     = info.trade_tick_size
    tick_value    = info.trade_tick_value
    swap_mode     = info.swap_mode
    swap_long_r   = info.swap_long
    swap_short_r  = info.swap_short
    rollover3days = info.swap_rollover3days

    comision_rt = obtener_comision_rt(symbol, datetime.now().year)
    mt5.shutdown()

    trades_long = trades_df_to_records(p, False) + trades_df_to_records(s, False)

    trades_short = trades_df_to_records(q, True) + trades_df_to_records(r, True)

    df_long, _, _ = _contabilizar(
        trades_long, CAPITAL_LONG, swap_long_r, swap_mode,
        contract_size, tick_value, tick_size, rollover3days, comision_rt
    )
    df_short, _, _ = _contabilizar(
        trades_short, CAPITAL_SHORT, swap_short_r, swap_mode,
        contract_size, tick_value, tick_size, rollover3days, comision_rt
    )

    imprimir_tabla([df_long], "LONG")
    imprimir_tabla([df_short], "SHORT")

    return df_long, df_short

if __name__ == "__main__":
    p, q, r, s = start_back()
    df_long, df_short = evaluar_realista(p, q, r, s)
