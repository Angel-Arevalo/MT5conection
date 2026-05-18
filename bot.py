import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import argparse
import sys
import os

from datetime import datetime, timedelta, timezone
from typing import Dict, Callable, Optional, Tuple, Any

import talib
import manager

sys.path.append(os.path.abspath('./optimal-moving-average'))

import keys
from find_best import opti_main

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", required=True)
parser.add_argument("--short", type=str, default="false")
args = parser.parse_args()

SYMBOL = str(args.symbol)
IS_SHORT = args.short.lower() in ["true", "1", "yes", "y"]

DIRECTION = "SHORT" if IS_SHORT else "LONG"
MAGIC_NUMBER = manager.get_or_create_magic(SYMBOL, DIRECTION)

BROKER_OFFSET = 3
N_REGIMEN_CONSEC = 2
N_CONFIRM = 2
ATR_PERIODS = 15

keys.calls = 15

#keys.methods = {"SMA", "EMA"}
#keys.candles = 1 
#keys.lookbacks = 3

FAST_METHODS: Dict[str, Callable] = {
    "SMA": talib.SMA,
    "EMA": talib.EMA,
    "WMA": talib.WMA,
    "DEMA": talib.DEMA,
    "TEMA": talib.TEMA,
    "TRIMA": talib.TRIMA,
    "KAMA": talib.KAMA,
    "T3": talib.T3,
    "MIDPOINT": talib.MIDPOINT,
}

def ts() -> str:
    return obtener_tiempo_servidor().strftime("%d/%m %H:%M:%S")

def obtener_tiempo_servidor() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=BROKER_OFFSET)

def obtener_filling_mode(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if info is None: return mt5.ORDER_FILLING_FOK
    mode = info.filling_mode
    if mode & 1: return mt5.ORDER_FILLING_FOK
    if mode & 2: return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def calcular_ma(close_arr: np.ndarray, metodo: str, lookback: int) -> np.ndarray:
    return FAST_METHODS[metodo](close_arr, timeperiod=lookback)

def obtener_señal(mid_arr: np.ndarray, ma: np.ndarray, is_short: bool) -> int:
    if len(ma) < 2 or np.isnan(ma[-1]) or np.isnan(ma[-2]):
        return 0

    if is_short:
        if ma[-2] >= mid_arr[-2] and ma[-1] < mid_arr[-1]: return -1
        if ma[-2] < mid_arr[-2] and ma[-1] >= mid_arr[-1]: return 1
    else:
        if ma[-2] <= mid_arr[-2] and ma[-1] > mid_arr[-1]: return 1
        if ma[-2] > mid_arr[-2] and ma[-1] <= mid_arr[-1]: return -1
    return 0

def estimar_ou(spread: np.ndarray) -> Tuple[Any, Any, Any]:
    S_prev = spread[:-1]
    S_curr = spread[1:]
    N = len(S_curr)
    if N < 3: return None, None, None

    Sx, Sy = S_prev.sum(), S_curr.sum()
    Sxx = (S_prev ** 2).sum()
    Sxy = (S_prev * S_curr).sum()
    Syy = (S_curr ** 2).sum()

    denom = N * Sxx - Sx ** 2
    if abs(denom) < 1e-14: return None, None, None

    beta = (N * Sxy - Sx * Sy) / denom
    if not (0 < beta < 1): return None, None, None

    theta = -np.log(beta)
    s2 = ((Syy - 2 * beta * Sxy + beta ** 2 * Sxx) / N) - (Sy / N - beta * Sx / N) ** 2
    if s2 <= 1e-14: return None, None, None

    e2t = np.exp(-2 * theta)
    den = 1 - e2t
    if den < 1e-14: return None, None, None

    sigma2 = s2 * 2 * theta / den
    v = sigma2 / (2 * theta) * den
    if v <= 1e-14: return None, None, None

    resid = S_curr - S_prev * beta
    logL = -N / 2 * np.log(2 * np.pi * v) - np.sum(resid ** 2) / (2 * v)
    return theta, np.sqrt(sigma2), logL

def estimar_gbm(spread: np.ndarray) -> Tuple[Any, Any, Any]:
    dS = np.diff(spread)
    N = len(dS)
    if N < 3: return None, None, None

    mu = dS.mean()
    sigma = dS.std(ddof=1)
    if sigma <= 1e-14: return None, None, None

    logL = -N / 2 * np.log(2 * np.pi * sigma ** 2) - np.sum((dS - mu) ** 2) / (2 * sigma ** 2)
    return mu, sigma, logL

def detectar_regimen(spread: np.ndarray, lookback: int, is_short: bool) -> str:
    valido = spread[~np.isnan(spread)]
    if len(valido) < lookback + 1: return "INDETERMINADO"

    ventana = valido[-(lookback + 1):]
    _, _, logL_OU = estimar_ou(ventana)
    mu, _, logL_GBM = estimar_gbm(ventana)

    if logL_OU is None or mu is None or logL_GBM is None: return "INDETERMINADO"
    if logL_OU - logL_GBM > 0: return "REVERSION"

    if is_short:
        return "ALCISTA_GBM" if mu > 0 else "BAJISTA_GBM"
    else:
        return "BAJISTA_GBM" if mu < 0 else "ALCISTA_GBM"

def obtener_data_optimizacion(symbol: str) -> Optional[pd.DataFrame]:
    ahora = obtener_tiempo_servidor()
    lunes_actual = (ahora - timedelta(days=ahora.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    viernes_fin = (lunes_actual - timedelta(days=3)).replace(hour=23, minute=59, second=59)
    lunes_inicio = lunes_actual - timedelta(weeks=4)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, lunes_inicio, viernes_fin)
    if rates is None or len(rates) == 0: return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    point = mt5.symbol_info(symbol).point
    df['bid'] = df['close'].astype(float)
    df['ask'] = (df['close'] + df['spread'] * point).astype(float)
    return df.set_index('time')

def inicializar_cache_velas(symbol: str, vela_min: int, max_elementos: int) -> Tuple[dict, int]:

    total_m1 = max_elementos * vela_min
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, total_m1)
    if rates is None or len(rates) == 0:
        return {'time': [], 'bid': [], 'ask': [], 'mid': []}, 0

    point = mt5.symbol_info(symbol).point
    vela_seg = vela_min * 60
    grouped = {}

    for r in rates:
        m1_time = int(r['time'])
        custom_open = (m1_time // vela_seg) * vela_seg
        bid = float(r['close'])
        ask = bid + float(r['spread']) * point
        grouped[custom_open] = (bid, ask)

    cache = {'time': [], 'bid': [], 'ask': [], 'mid': []}
    for t in sorted(grouped.keys()):
        bid, ask = grouped[t]
        cache['time'].append(t)
        cache['bid'].append(bid)
        cache['ask'].append(ask)
        cache['mid'].append((bid + ask) / 2.0)

    return cache, int(rates[-1]['time'])

def actualizar_cache_velas(symbol: str, vela_min: int, cache: dict, last_m1_time: int, max_elementos: int) -> int:

    ahora_ts = int(obtener_tiempo_servidor().timestamp()) + 3600 # Margen de seguridad hacia el futuro

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, last_m1_time, ahora_ts)

    if rates is None or len(rates) == 0:
        return last_m1_time

    point = mt5.symbol_info(symbol).point
    vela_seg = vela_min * 60

    for r in rates:
        m1_time = int(r['time'])
        custom_open = (m1_time // vela_seg) * vela_seg
        bid = float(r['close'])
        ask = bid + float(r['spread']) * point
        mid = (bid + ask) / 2.0

        if len(cache['time']) > 0 and cache['time'][-1] == custom_open:

            cache['bid'][-1] = bid
            cache['ask'][-1] = ask
            cache['mid'][-1] = mid
        else:

            cache['time'].append(custom_open)
            cache['bid'].append(bid)
            cache['ask'].append(ask)
            cache['mid'].append(mid)

    if len(cache['time']) > max_elementos:
        corte = len(cache['time']) - max_elementos
        cache['time'] = cache['time'][corte:]
        cache['bid'] = cache['bid'][corte:]
        cache['ask'] = cache['ask'][corte:]
        cache['mid'] = cache['mid'][corte:]

    return int(rates[-1]['time'])

def ejecutar_orden(tipo: int, comentario: str) -> bool:
    filling = obtener_filling_mode(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None: return False

    if ((not IS_SHORT and tipo == mt5.ORDER_TYPE_BUY) or (IS_SHORT and tipo == mt5.ORDER_TYPE_SELL)):
        volumen = manager.calcular_volumen_estricto(SYMBOL, MAGIC_NUMBER, DIRECTION)
        if volumen <= 0: return False

        precio = tick.ask if not IS_SHORT else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": float(volumen),
            "type": tipo, "price": float(precio), "magic": int(MAGIC_NUMBER),
            "comment": comentario, "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling,
        }
        res = mt5.order_send(req)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    posiciones = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if posiciones is None: return False

    ok = True
    for p in posiciones:
        precio = mt5.symbol_info_tick(SYMBOL).bid if not IS_SHORT else mt5.symbol_info_tick(SYMBOL).ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": float(p.volume),
            "type": tipo, "position": int(p.ticket), "price": float(precio),
            "magic": int(MAGIC_NUMBER), "comment": comentario, "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling,
        }
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE: ok = False
    return ok

def esperar_hasta_siguiente_vela(vela_min: int) -> None:
    ahora = obtener_tiempo_servidor()
    vela_seg = vela_min * 60
    now_ts = int(ahora.timestamp())
    next_ts = ((now_ts // vela_seg) + 1) * vela_seg
    sleep_s = next_ts - now_ts + 1
    print(f"[{ts()}]: Durmiendo {sleep_s} segundos.")
    time.sleep(max(1, sleep_s))

def esperar_hasta_lunes():
    ahora = obtener_tiempo_servidor()
    dias = (7 - ahora.weekday()) % 7
    if dias == 0: dias = 7
    prox = (ahora + timedelta(days=dias)).replace(hour=0, minute=0, second=0, microsecond=0)
    secs = (prox - ahora).total_seconds()
    print(f"[{ts()}]: Durmiendo hasta el lunes, {secs:.0f} segundos.")
    time.sleep(max(1, secs))

def main():
    if not mt5.initialize(): return

    ORDER_OPEN = mt5.ORDER_TYPE_SELL if IS_SHORT else mt5.ORDER_TYPE_BUY
    ORDER_CLOSE = mt5.ORDER_TYPE_BUY if IS_SHORT else mt5.ORDER_TYPE_SELL
    SIGNAL_OPEN = -1 if IS_SHORT else 1
    SIGNAL_CLOSE = 1 if IS_SHORT else -1

    while True:
        data_opt = obtener_data_optimizacion(SYMBOL)
        if data_opt is None:
            time.sleep(60)
            continue

        params = opti_main(data_opt[['bid', 'ask']], is_bid=True, shorts=IS_SHORT)
        metodo_ma = params[0]
        vela_min = int(params[1])
        lookback = int(params[2])

        max_elementos = lookback + 150
        max_espera = max(lookback // 2, N_REGIMEN_CONSEC + N_CONFIRM + 1)

        print(f"[{ts()}] {SYMBOL} {'SHORT' if IS_SHORT else 'LONG'} {metodo_ma} {vela_min}m LB={lookback}")

        velas_cache, last_m1_time = inicializar_cache_velas(SYMBOL, vela_min, max_elementos)

        posiciones = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        posicion_abierta = posiciones is not None and len(posiciones) > 0

        en_señal = False
        regimen_confirmado = False
        velas_regimen = 0; velas_reversion = 0; velas_espera = 0
        last_bar = None

        while True:
            ahora = obtener_tiempo_servidor()
            if ahora.weekday() == 4 and ahora.hour == 23 and ahora.minute >= 50:
                if posicion_abierta:
                    ejecutar_orden(ORDER_CLOSE, "Cierre Viernes")
                    posicion_abierta = False
                esperar_hasta_lunes()
                break

            esperar_hasta_siguiente_vela(vela_min)

            last_m1_time = actualizar_cache_velas(SYMBOL, vela_min, velas_cache, last_m1_time, max_elementos)

            if len(velas_cache['time']) < lookback + 5: continue

            current_bar = velas_cache['time'][-1]
            if last_bar == current_bar: continue
            last_bar = current_bar

            mid_arr = np.array(velas_cache['mid'])[:-1]
            bid_arr = np.array(velas_cache['bid'])[:-1]
            ask_arr = np.array(velas_cache['ask'])[:-1]

            precio_bid = float(bid_arr[-1])
            precio_ask = float(ask_arr[-1])

            ma_arr = calcular_ma(mid_arr, metodo_ma, lookback)
            señal = obtener_señal(mid_arr, ma_arr, IS_SHORT)
 
            print(señal)

            if señal == SIGNAL_CLOSE and posicion_abierta:
                if ejecutar_orden(ORDER_CLOSE, "Cierre Señal"):
                    posicion_abierta = False
                    en_señal = False
                    regimen_confirmado = False
                    velas_regimen = 0; velas_reversion = 0; velas_espera = 0
                    print(f"[{ts()}] CLOSE {precio_bid:.5f}")

            elif señal == SIGNAL_CLOSE and not posicion_abierta:
                en_señal = False
                regimen_confirmado = False
                velas_regimen = 0; velas_reversion = 0; velas_espera = 0

            elif señal == SIGNAL_OPEN and not posicion_abierta and not en_señal:
                en_señal = True
                regimen_confirmado = False
                velas_regimen = 0; velas_reversion = 0; velas_espera = 0
                print(f"[{ts()}] SIGNAL {precio_bid:.5f}")

            if en_señal and not posicion_abierta:

                regimen = detectar_regimen(mid_arr - ma_arr, lookback, IS_SHORT)
                velas_espera += 1

                if IS_SHORT:
                    if regimen == "ALCISTA_GBM":
                        velas_regimen += 1
                        velas_reversion = 0
                        if velas_regimen >= N_REGIMEN_CONSEC: regimen_confirmado = True
                    else:
                        velas_regimen = 0
                        if regimen_confirmado:
                            if regimen == "REVERSION":
                                velas_reversion += 1
                                if velas_reversion >= N_CONFIRM:
                                    if ejecutar_orden(ORDER_OPEN, "Entrada SHORT"):
                                        posicion_abierta = True
                                        en_señal = False
                                        regimen_confirmado = False
                                        velas_regimen = 0; velas_reversion = 0; velas_espera = 0
                                        print(f"[{ts()}] SHORT {precio_bid:.5f}")
                            else:
                                velas_reversion = 0
                        else:
                            if ejecutar_orden(ORDER_OPEN, "Entrada Directa SHORT"):
                                posicion_abierta = True
                                en_señal = False
                                velas_regimen = 0; velas_reversion = 0; velas_espera = 0
                                print(f"[{ts()}] SHORT DIRECT {precio_bid:.5f}")
                else:
                    if regimen == "BAJISTA_GBM":
                        velas_regimen += 1
                        velas_reversion = 0
                        if velas_regimen >= N_REGIMEN_CONSEC: regimen_confirmado = True
                    else:
                        velas_regimen = 0
                        if regimen_confirmado:
                            velas_reversion += 1
                            if velas_reversion >= N_CONFIRM:
                                if ejecutar_orden(ORDER_OPEN, "Entrada LONG"):
                                    posicion_abierta = True
                                    en_señal = False
                                    regimen_confirmado = False
                                    velas_regimen = 0; velas_reversion = 0; velas_espera = 0
                                    print(f"[{ts()}] LONG {precio_bid:.5f}")
                        else:
                            if ejecutar_orden(ORDER_OPEN, "Entrada Directa LONG"):
                                posicion_abierta = True
                                en_señal = False
                                velas_regimen = 0; velas_reversion = 0; velas_espera = 0
                                print(f"[{ts()}] LONG DIRECT {precio_bid:.5f}")

                if en_señal and velas_espera >= max_espera:
                    en_señal = False
                    regimen_confirmado = False
                    velas_regimen = 0; velas_reversion = 0; velas_espera = 0
                    print(f"[{ts()}] Cancelada por espera")

if __name__ == "__main__":
    main()
