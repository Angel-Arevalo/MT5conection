import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import argparse
import sys
import os

from datetime import datetime, timedelta, timezone
from typing import Dict, Callable, Optional, Tuple

import talib
import manager

sys.path.append(os.path.abspath('./optimal-moving-average'))

import keys
from find_best import opti_main

parser = argparse.ArgumentParser()
parser.add_argument("--symbol",   required=True)
parser.add_argument("--short",    type=str, default="false")
parser.add_argument("--mon",      type=int, default=1000)
parser.add_argument("--st_mon",   type=str, default="false")
parser.add_argument("--leverage", type=int, default=40)
args = parser.parse_args()

SYMBOL      = str(args.symbol)
IS_SHORT    = args.short.lower() in ["true", "1", "yes", "y"]
MON         = args.mon
STATIC_MON  = args.st_mon.lower() in ["true", "1", "yes", "y"]
LEVERAGE    = args.leverage

if LEVERAGE <= 0 or LEVERAGE > 400 or not isinstance(LEVERAGE, int):
    raise ValueError("Valor de apalancamiento no permitido")

DIRECTION    = "SHORT" if IS_SHORT else "LONG"
MAGIC_NUMBER = manager.get_or_create_magic(SYMBOL, DIRECTION)

N_ADX_CONSEC    = 2
N_CONFIRM       = 2
ADX_THRESHOLD   = 25

keys.calls   = 15

FAST_METHODS: Dict[str, Callable] = {
    "SMA": talib.SMA, "EMA": talib.EMA, "WMA": talib.WMA,
    "DEMA": talib.DEMA, "TEMA": talib.TEMA, "TRIMA": talib.TRIMA,
    "KAMA": talib.KAMA, "T3": talib.T3, "MIDPOINT": talib.MIDPOINT,
}

_LAST_TICK_TIME = 0
_BROKER_OFFSET_HOURS = 0.0


def comprobar_conexion() -> bool:

    terminal = mt5.terminal_info()
    if terminal is None:
        return False
    return terminal.connected

def asegurar_conexion():

    if comprobar_conexion():
        return True

    print(f"[{datetime.now().strftime('%d/%m %H:%M:%S')}] ⚠️ ¡CONEXIÓN PERDIDA! Detectada caída de Wi-Fi o Broker.")

    while not comprobar_conexion():
        print(f"[{datetime.now().strftime('%d/%m %H:%M:%S')}] Intentando reconectar a MetaTrader 5...")
        mt5.shutdown()
        time.sleep(5)

        if mt5.initialize():
            time.sleep(5)
            if comprobar_conexion():
                print(f"[{datetime.now().strftime('%d/%m %H:%M:%S')}] ✅ ¡Reconexión exitosa con el Broker!")
                return True
        time.sleep(5)


def obtener_tiempo_servidor() -> datetime:
    global _LAST_TICK_TIME, _BROKER_OFFSET_HOURS

    asegurar_conexion()

    tick = mt5.symbol_info_tick("US500_SPOT")

    if tick is None:
        return datetime.now(timezone.utc) + timedelta(hours=_BROKER_OFFSET_HOURS)

    if _BROKER_OFFSET_HOURS == 0:
        if tick.time != _LAST_TICK_TIME:
            _LAST_TICK_TIME = tick.time
            pc_utc_ts = datetime.now(timezone.utc).timestamp()
            diff_seconds = round(tick.time - pc_utc_ts)
            _BROKER_OFFSET_HOURS = diff_seconds / 3600.0

    return datetime.now(timezone.utc) + timedelta(hours=_BROKER_OFFSET_HOURS)


def ts() -> str:
    return obtener_tiempo_servidor().strftime("%d/%m %H:%M:%S")


def calcular_ma(arr: np.ndarray, metodo: str, lb: int) -> np.ndarray:
    return FAST_METHODS[metodo](arr, timeperiod=lb)

def obtener_señal(mid_arr: np.ndarray, ma: np.ndarray, is_short: bool) -> int:
    if len(ma) < 2 or np.isnan(ma[-1]) or np.isnan(ma[-2]):
        return 0

    if is_short:
        if ma[-2] >= mid_arr[-2] and ma[-1] < mid_arr[-1]:  return -1
        if ma[-2] < mid_arr[-2]  and ma[-1] >= mid_arr[-1]: return  1
    else:
        if ma[-2] <= mid_arr[-2] and ma[-1] > mid_arr[-1]:  return  1
        if ma[-2] > mid_arr[-2]  and ma[-1] <= mid_arr[-1]: return -1
    return 0

def adx_favorable(high: np.ndarray, low: np.ndarray, close: np.ndarray, lb: int) -> bool:
    if len(close) < lb + 2:
        return True

    adx      = talib.ADX(high, low, close, timeperiod=lb)
    adx_cur  = adx[-1]
    adx_prev = adx[-2]

    if np.isnan(adx_cur) or np.isnan(adx_prev):
        return True

    if adx_cur < ADX_THRESHOLD:
        return True

    if adx_cur > ADX_THRESHOLD and adx_cur < adx_prev:
        return True

    return False  


def inicializar_cache_velas(symbol: str, vela_min: int, max_elementos: int) -> Tuple[dict, int]:
    asegurar_conexion()
    total_m1 = max_elementos * vela_min
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, total_m1)
    if rates is None or len(rates) == 0:
        return _cache_vacio(), 0

    point    = mt5.symbol_info(symbol).point
    vela_seg = vela_min * 60
    grouped  = {}

    for r in rates:
        m1_time    = int(r['time'])
        custom_open = (m1_time // vela_seg) * vela_seg
        bid  = float(r['close'])
        ask  = bid + float(r['spread']) * point
        high = float(r['high'])
        low  = float(r['low'])

        if custom_open in grouped:
            prev = grouped[custom_open]
            grouped[custom_open] = [bid, ask, max(prev[2], high), min(prev[3], low)]
        else:
            grouped[custom_open] = [bid, ask, high, low]

    cache = _cache_vacio()
    for t in sorted(grouped.keys()):
        bid, ask, high, low = grouped[t]
        mid = (bid + ask) / 2.0
        cache['time'].append(t)
        cache['bid'].append(bid)
        cache['ask'].append(ask)
        cache['mid'].append(mid)
        cache['high'].append(high)
        cache['low'].append(low)

    return cache, int(rates[-1]['time'])


def actualizar_cache_velas(symbol: str, vela_min: int, cache: dict,
                           last_m1_time: int, max_elementos: int) -> int:
    asegurar_conexion()
    ahora_ts = int(obtener_tiempo_servidor().timestamp()) + 3600
    rates    = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, last_m1_time, ahora_ts)
    if rates is None or len(rates) == 0:
        return last_m1_time

    point    = mt5.symbol_info(symbol).point
    vela_seg = vela_min * 60

    for r in rates:
        m1_time    = int(r['time'])
        custom_open = (m1_time // vela_seg) * vela_seg
        bid  = float(r['close'])
        ask  = bid + float(r['spread']) * point
        mid  = (bid + ask) / 2.0
        high = float(r['high'])
        low  = float(r['low'])

        if len(cache['time']) > 0 and cache['time'][-1] == custom_open:
            cache['bid'][-1]  = bid
            cache['ask'][-1]  = ask
            cache['mid'][-1]  = mid
            cache['high'][-1] = max(cache['high'][-1], high)
            cache['low'][-1]  = min(cache['low'][-1],  low)
        else:
            cache['time'].append(custom_open)
            cache['bid'].append(bid)
            cache['ask'].append(ask)
            cache['mid'].append(mid)
            cache['high'].append(high)
            cache['low'].append(low)

    if len(cache['time']) > max_elementos:
        corte = len(cache['time']) - max_elementos
        for k in cache:
            cache[k] = cache[k][corte:]

    return int(rates[-1]['time'])


def _cache_vacio() -> dict:
    return {'time': [], 'bid': [], 'ask': [], 'mid': [], 'high': [], 'low': []}


def obtener_filling_mode(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if info is None: return mt5.ORDER_FILLING_FOK
    mode = info.filling_mode
    if mode & 1: return mt5.ORDER_FILLING_FOK
    if mode & 2: return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def ejecutar_orden(tipo: int, comentario: str) -> bool:
    asegurar_conexion()
    filling = obtener_filling_mode(SYMBOL)
    tick    = mt5.symbol_info_tick(SYMBOL)
    if tick is None: return False

    es_apertura = (
        (not IS_SHORT and tipo == mt5.ORDER_TYPE_BUY) or
        (IS_SHORT      and tipo == mt5.ORDER_TYPE_SELL)
    )

    if es_apertura:
        volumen = manager.calcular_volumen_estricto(
            SYMBOL, MAGIC_NUMBER, DIRECTION,
            capital=MON, apalancamiento=LEVERAGE,
            ignorar_historial=STATIC_MON
        )
        if volumen <= 0: return False

        precio = tick.ask if not IS_SHORT else tick.bid
        req = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      SYMBOL,
            "volume":      float(volumen),
            "type":        tipo,
            "price":       float(precio),
            "magic":       int(MAGIC_NUMBER),
            "comment":     comentario,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        res = mt5.order_send(req)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    posiciones = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if posiciones is None: return False

    ok = True
    for p in posiciones:
        tick_close = mt5.symbol_info_tick(SYMBOL)
        if tick_close is None: return False
        precio = tick_close.bid if not IS_SHORT else tick_close.ask
        req = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      SYMBOL,
            "volume":      float(p.volume),
            "type":        tipo,
            "position":    int(p.ticket),
            "price":       float(precio),
            "magic":       int(MAGIC_NUMBER),
            "comment":     comentario,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            ok = False
    return ok


def esperar_hasta_siguiente_vela(vela_min: int) -> None:
    ahora    = obtener_tiempo_servidor()
    vela_seg = vela_min * 60
    now_ts   = int(ahora.timestamp())
    next_ts  = ((now_ts // vela_seg) + 1) * vela_seg
    sleep_s  = next_ts - now_ts + 2
    print(f"[{ts()}] Durmiendo {sleep_s}s hasta próxima vela.")
    time.sleep(max(1, sleep_s))

def esperar_hasta_lunes() -> None:
    ahora = obtener_tiempo_servidor()
    dias  = (7 - ahora.weekday()) % 7
    if dias == 0: dias = 7
    prox  = (ahora + timedelta(days=dias)).replace(hour=0, minute=0, second=0, microsecond=0)
    secs  = (prox - ahora).total_seconds()
    print(f"[{ts()}] Fin de semana. Durmiendo {secs:.0f}s hasta el lunes.")
    time.sleep(max(1, secs))

def obtener_data_optimizacion(symbol: str) -> Optional[pd.DataFrame]:
    asegurar_conexion()
    ahora       = obtener_tiempo_servidor()
    lunes_actual = (ahora - timedelta(days=ahora.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0)
    viernes_fin  = (lunes_actual - timedelta(days=3)).replace(hour=23, minute=59, second=59)
    lunes_inicio = lunes_actual - timedelta(weeks=4)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, lunes_inicio, viernes_fin)
    if rates is None or len(rates) == 0: return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    point = mt5.symbol_info(symbol).point
    df['bid'] = df['close'].astype(float)
    df['ask'] = (df['close'] + df['spread'] * point).astype(float)
    return df.set_index('time')


def main():
    if not mt5.initialize(): return

    ORDER_OPEN  = mt5.ORDER_TYPE_SELL if IS_SHORT else mt5.ORDER_TYPE_BUY
    ORDER_CLOSE = mt5.ORDER_TYPE_BUY  if IS_SHORT else mt5.ORDER_TYPE_SELL
    SIGNAL_OPEN  = -1 if IS_SHORT else  1
    SIGNAL_CLOSE =  1 if IS_SHORT else -1

    while True:
        data_opt = obtener_data_optimizacion(SYMBOL)
        if data_opt is None:
            print(f"[{ts()}] Sin datos de optimización, reintentando en 60s.")
            time.sleep(60)
            continue

        params     = opti_main(data_opt[['bid', 'ask']], is_bid=True,
                               verbose=True, shorts=IS_SHORT)
        metodo_ma  = params[0]
        vela_min   = int(params[1])
        lookback   = int(params[2])

        max_elementos = lookback + 150
        max_espera    = max(lookback // 2, N_ADX_CONSEC + N_CONFIRM + 1)

        print(f"[{ts()}] {SYMBOL} {DIRECTION} | {metodo_ma} {vela_min}m LB={lookback} "
              f"| ADX umbral={ADX_THRESHOLD}")

        velas_cache, last_m1_time = inicializar_cache_velas(SYMBOL, vela_min, max_elementos)

        posiciones       = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        posicion_abierta = posiciones is not None and len(posiciones) > 0

        en_señal         = False
        adx_confirmado   = False
        velas_adx        = 0
        velas_confirm    = 0
        velas_espera     = 0
        last_bar         = None

        while True:
            ahora = obtener_tiempo_servidor()

            if ahora.weekday() == 4 and ahora.hour == 23 and ahora.minute >= 50:
                posiones_check = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
                if posiones_check is not None and len(posiones_check) > 0:
                    ejecutar_orden(ORDER_CLOSE, "Cierre Viernes")
                posicion_abierta = False
                esperar_hasta_lunes()
                break

            esperar_hasta_siguiente_vela(vela_min)

            asegurar_conexion()
            posiones_actuales = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)

            if posiones_actuales is None:

                if not comprobar_conexion(): continue
                realmente_abierta = False
            else:
                realmente_abierta = len(posiones_actuales) > 0

            if posicion_abierta and not realmente_abierta:
                print(f"[{ts()}] ℹ️ Sincronización: La posición se cerró externamente (posible SL/TP durante corte).")
                posicion_abierta = False
                en_señal = adx_confirmado = False
                velas_adx = velas_confirm = velas_espera = 0
            elif not posicion_abierta and realmente_abierta:
                print(f"[{ts()}] ℹ️ Sincronización: Se detectó posición abierta huérfana en el Broker. Tomando control.")
                posicion_abierta = True


            last_m1_time = actualizar_cache_velas(
                SYMBOL, vela_min, velas_cache, last_m1_time, max_elementos)

            if len(velas_cache['time']) < lookback + 5:
                print(f"[{ts()}] Velas insuficientes, esperando.")
                continue

            current_bar = velas_cache['time'][-1]
            while last_bar == current_bar:
                print(f"[{ts()}] Sin vela nueva, reintentando en 0.5s.")
                time.sleep(0.5)
                last_m1_time = actualizar_cache_velas(
                    SYMBOL, vela_min, velas_cache, last_m1_time, max_elementos)
                current_bar = velas_cache['time'][-1]

            last_bar = current_bar

            mid_arr  = np.array(velas_cache['mid'])[:-1]
            bid_arr  = np.array(velas_cache['bid'])[:-1]
            ask_arr  = np.array(velas_cache['ask'])[:-1]
            high_arr = np.array(velas_cache['high'])[:-1]
            low_arr  = np.array(velas_cache['low'])[:-1]

            precio_bid = float(bid_arr[-1])
            precio_ask = float(ask_arr[-1])

            ma_arr = calcular_ma(mid_arr, metodo_ma, lookback)
            señal  = obtener_señal(mid_arr, ma_arr, IS_SHORT)

            print(f"[{ts()}] señal={señal} bid={precio_bid:.5f} "
                  f"adx_conf={adx_confirmado} v_adx={velas_adx} v_rev={velas_confirm}")

            if señal == SIGNAL_CLOSE and posicion_abierta:
                if ejecutar_orden(ORDER_CLOSE, "Cierre Señal"):
                    posicion_abierta = False
                    en_señal = adx_confirmado = False
                    velas_adx = velas_confirm = velas_espera = 0
                    print(f"[{ts()}] CLOSE @ {precio_bid:.5f}")

            elif señal == SIGNAL_CLOSE and not posicion_abierta:
                en_señal = adx_confirmado = False
                velas_adx = velas_confirm = velas_espera = 0

            elif señal == SIGNAL_OPEN and not posicion_abierta and not en_señal:
                en_señal = True
                adx_confirmado = False
                velas_adx = velas_confirm = velas_espera = 0
                print(f"[{ts()}] SIGNAL @ {precio_bid:.5f}")

            if en_señal and not posicion_abierta:
                velas_espera += 1
                favorable = adx_favorable(high_arr, low_arr, mid_arr, lookback)

                if not favorable:
                    en_señal = adx_confirmado = False
                    velas_adx = velas_confirm = velas_espera = 0
                    print(f"[{ts()}] ADX bloqueó entrada — tendencia fuerte")
                else:
                    velas_adx += 1
                    if velas_adx >= N_ADX_CONSEC:
                        adx_confirmado = True

                    if adx_confirmado:
                        velas_confirm += 1
                        if velas_confirm >= N_CONFIRM:
                            if ejecutar_orden(ORDER_OPEN, f"Entrada {DIRECTION}"):
                                posicion_abierta = True
                                en_señal = adx_confirmado = False
                                velas_adx = velas_confirm = velas_espera = 0
                                print(f"[{ts()}] {DIRECTION} confirmado @ {precio_ask:.5f}")
                    else:
                        if ejecutar_orden(ORDER_OPEN, f"Entrada Directa {DIRECTION}"):
                            posicion_abierta = True
                            en_señal = False
                            velas_adx = velas_confirm = velas_espera = 0
                            print(f"[{ts()}] {DIRECTION} directo @ {precio_ask:.5f}")

                if en_señal and velas_espera >= max_espera:
                    en_señal = adx_confirmado = False
                    velas_adx = velas_confirm = velas_espera = 0
                    print(f"[{ts()}] Señal cancelada por timeout ({max_espera} velas)")


if __name__ == "__main__":
    main()
