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
parser.add_argument("--symbol", required=True)
parser.add_argument("--short", type=str, default="false")
parser.add_argument("--mon", type=int, default=1000)
parser.add_argument("--st_mon", type=str, default="false")
parser.add_argument("--leverage", type=int, default=40)
args = parser.parse_args()

SYMBOL = str(args.symbol)
IS_SHORT = args.short.lower() in ["true", "1", "yes", "y"]
MON = args.mon
STATIC_MON = args.st_mon.lower() in ["true", "1", "yes", "y"]
LEVERAGE = args.leverage

if LEVERAGE <= 0 or LEVERAGE > 400 or not isinstance(LEVERAGE, int):
    raise ValueError("Apalancamiento no valido")

DIRECTION = "SHORT" if IS_SHORT else "LONG"
MAGIC_NUMBER = manager.get_or_create_magic(SYMBOL, DIRECTION)
keys.calls = 15
keys.methods = {"SMA"}
keys.lookbacks = 3


FAST_METHODS: Dict[str, Callable] = {
    "SMA": talib.SMA, "EMA": talib.EMA, "WMA": talib.WMA,
    "DEMA": talib.DEMA, "TEMA": talib.TEMA, "TRIMA": talib.TRIMA,
    "KAMA": talib.KAMA, "T3": talib.T3, "MIDPOINT": talib.MIDPOINT,
}

_LAST_TICK_TIME = 0
_BROKER_OFFSET_HOURS = 0.0

def comprobar_conexion() -> bool:
    terminal = mt5.terminal_info()
    return terminal is not None and terminal.connected

def asegurar_conexion():
    if comprobar_conexion():
        return True
    print(f"[{datetime.now().strftime('%d/%m %H:%M:%S')}] Conexion perdida. Reconectando...")
    while not comprobar_conexion():
        mt5.shutdown()
        time.sleep(5)
        if mt5.initialize():
            time.sleep(2)
            if comprobar_conexion():
                print(f"[{datetime.now().strftime('%d/%m %H:%M:%S')}] Reconexion exitosa")
                return True
        time.sleep(3)

def obtener_tiempo_servidor() -> datetime:
    global _LAST_TICK_TIME, _BROKER_OFFSET_HOURS
    asegurar_conexion()
    tick = mt5.symbol_info_tick(SYMBOL)
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

def encontrar_ruta_pre_calc(symbol: str) -> str:
    for root, dirs, files in os.walk('.'):
        if 'back_normal_reversion.py' in files:
            ruta_dir = os.path.join(root, 'pre_calc')
            os.makedirs(ruta_dir, exist_ok=True)
            return os.path.join(ruta_dir, f"{symbol}.csv")
    ruta_dir = os.path.join('.', 'pre_calc')
    os.makedirs(ruta_dir, exist_ok=True)
    return os.path.join(ruta_dir, f"{symbol}.csv")

def _cargar_pre_calc(ruta: str) -> dict:
    if not os.path.exists(ruta):
        return {}
    try:
        df = pd.read_csv(ruta, dtype={'semana': str, 'met_l': str, 'met_s': str})
        cache = {}
        for _, row in df.iterrows():
            cache[row['semana']] = {
                'met_l': row['met_l'],
                'vela_l': int(row['vela_l']),
                'lb_l': int(row['lb_l']),
                'met_s': row['met_s'],
                'vela_s': int(row['vela_s']),
                'lb_s': int(row['lb_s']),
            }
        return cache
    except Exception:
        return {}

def _guardar_pre_calc(ruta: str, cache: dict) -> None:
    filas = [
        {
            'semana': semana,
            'met_l': v['met_l'], 'vela_l': v['vela_l'], 'lb_l': v['lb_l'],
            'met_s': v['met_s'], 'vela_s': v['vela_s'], 'lb_s': v['lb_s'],
        }
        for semana, v in sorted(cache.items())
    ]
    pd.DataFrame(filas).to_csv(ruta, index=False)

def verificar_posicion_activa() -> bool:
    asegurar_conexion()
    posiciones = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if posiciones is None or len(posiciones) == 0:
        return False
    tipo_esperado = mt5.POSITION_TYPE_SELL if IS_SHORT else mt5.POSITION_TYPE_BUY
    for pos in posiciones:
        if pos.symbol == SYMBOL and pos.magic == MAGIC_NUMBER and pos.type == tipo_esperado:
            return True
    return False

def obtener_data_optimizacion(symbol: str) -> Optional[pd.DataFrame]:
    asegurar_conexion()
    ahora = obtener_tiempo_servidor()
    lunes_actual = (ahora - timedelta(days=ahora.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    viernes_fin = (lunes_actual - timedelta(days=3)).replace(hour=23, minute=59, second=59)
    lunes_inicio = lunes_actual - timedelta(weeks=4)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, lunes_inicio, viernes_fin)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    point = mt5.symbol_info(symbol).point
    df['bid'] = df['close'].astype(float)
    df['ask'] = (df['close'] + df['spread'] * point).astype(float)
    return df.set_index('time')

def verificar_y_optimizar_semana(symbol: str) -> Tuple[str, int, int]:
    ruta_csv = encontrar_ruta_pre_calc(symbol)
    cache_pre = _cargar_pre_calc(ruta_csv)
    ahora = obtener_tiempo_servidor()
    lunes_actual = (ahora - timedelta(days=ahora.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    semana_key = str(lunes_actual.date())
    if semana_key in cache_pre:
        vals = cache_pre[semana_key]
        print(f"[{ts()}] Parametros cargados de cache para la semana {semana_key}")
        if IS_SHORT:
            return vals['met_s'], int(vals['vela_s']), int(vals['lb_s'])
        else:
            return vals['met_l'], int(vals['vela_l']), int(vals['lb_l'])
    print(f"[{ts()}] Optimizando para la semana {semana_key}")
    data_opt = obtener_data_optimizacion(symbol)
    if data_opt is None or data_opt.empty:
        raise RuntimeError("Datos insuficientes para optimizar")
    p_l = opti_main(data_opt[['bid', 'ask']], is_bid=True, verbose=False, shorts=False)
    p_s = opti_main(data_opt[['bid', 'ask']], is_bid=True, verbose=False, shorts=True)
    cache_pre[semana_key] = {
        'met_l': p_l[0], 'vela_l': int(p_l[1]), 'lb_l': int(p_l[2]),
        'met_s': p_s[0], 'vela_s': int(p_s[1]), 'lb_s': int(p_s[2]),
    }
    _guardar_pre_calc(ruta_csv, cache_pre)
    print(f"[{ts()}] Optimizacion guardada")
    vals = cache_pre[semana_key]
    if IS_SHORT:
        return vals['met_s'], int(vals['vela_s']), int(vals['lb_s'])
    else:
        return vals['met_l'], int(vals['vela_l']), int(vals['lb_l'])

def inicializar_cache_velas(symbol: str, vela_min: int, max_elementos: int) -> Tuple[dict, int]:
    asegurar_conexion()
    total_m1 = max_elementos * vela_min
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, total_m1)
    if rates is None or len(rates) == 0:
        return _cache_vacio(), 0
    point = mt5.symbol_info(symbol).point
    vela_seg = vela_min * 60
    grouped = {}
    for r in rates:
        m1_time = int(r['time'])
        custom_open = (m1_time // vela_seg) * vela_seg
        bid = float(r['close'])
        ask = bid + float(r['spread']) * point
        grouped[custom_open] = [bid, ask]
    cache = _cache_vacio()
    for t in sorted(grouped.keys()):
        bid, ask = grouped[t]
        cache['time'].append(t)
        cache['bid'].append(bid)
        cache['ask'].append(ask)
        cache['mid'].append((bid + ask) / 2.0)
    return cache, int(rates[-1]['time'])

def actualizar_cache_velas(symbol: str, vela_min: int, cache: dict, last_m1_time: int, max_elementos: int) -> int:
    asegurar_conexion()
    ahora_ts = int(obtener_tiempo_servidor().timestamp()) + 3600
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

        if len(cache['time']) > 0 and cache['time'][-1] == custom_open:
            cache['bid'][-1] = bid
            cache['ask'][-1] = ask

        else:
            cache['time'].append(custom_open)
            cache['bid'].append(bid)
            cache['ask'].append(ask)

    if len(cache['time']) > max_elementos:
        corte = len(cache['time']) - max_elementos
        for k in cache:
            cache[k] = cache[k][corte:]
    return int(rates[-1]['time'])

def _cache_vacio() -> dict:
    return {'time': [], 'bid': [], 'ask': [], 'mid': []}

def evaluar_señal_mr(arr: np.ndarray, metodo: str, lb: int) -> int:
    ma = FAST_METHODS[metodo](arr, timeperiod=lb)

    if len(ma) < 2 or np.isnan(ma[-1]) or np.isnan(ma[-2]):
        return 0
    if arr[-2] >= ma[-2] and arr[-1] < ma[-1]:
        return 1
    if arr[-2] <= ma[-2] and arr[-1] > ma[-1]:
        return -1
    return 0

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
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None: return False
    es_apertura = (
        (not IS_SHORT and tipo == mt5.ORDER_TYPE_BUY) or
        (IS_SHORT and tipo == mt5.ORDER_TYPE_SELL)
    )
    if es_apertura:
        volumen = manager.calcular_volumen_estricto(
            SYMBOL, MAGIC_NUMBER, DIRECTION,
            capital=MON, apalancamiento=LEVERAGE,
            ignorar_historial=STATIC_MON
        )
        if volumen <= 0: return False
        precio = tick.ask if tipo == mt5.ORDER_TYPE_BUY else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(volumen),
            "type": tipo,
            "price": float(precio),
            "magic": int(MAGIC_NUMBER),
            "comment": comentario,
            "type_time": mt5.ORDER_TIME_GTC,
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
        precio = tick_close.bid if p.type == mt5.POSITION_TYPE_BUY else tick_close.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(p.volume),
            "type": tipo,
            "position": int(p.ticket),
            "price": float(precio),
            "magic": int(MAGIC_NUMBER),
            "comment": comentario,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            ok = False
    return ok

def esperar_hasta_siguiente_vela(vela_min: int) -> None:
    ahora = obtener_tiempo_servidor()

    vela_seg = vela_min * 60
    now_ts = int(ahora.timestamp())
    next_ts = ((now_ts // vela_seg) + 1) * vela_seg

    sleep_s = next_ts - now_ts + 1
    print(f"[{ts()}] Esperando {sleep_s}s hasta fin de vela")
    time.sleep(max(1, sleep_s))

def esperar_hasta_lunes() -> None:
    ahora = obtener_tiempo_servidor()
    dias = (7 - ahora.weekday()) % 7
    if dias == 0: dias = 7
    prox = (ahora + timedelta(days=dias)).replace(hour=0, minute=0, second=0, microsecond=0)
    secs = (prox - ahora).total_seconds()
    print(f"[{ts()}] Mercado cerrado. Esperando al lunes")
    time.sleep(max(1, secs))

def main():
    if not mt5.initialize():
        print("Error inicializando MetaTrader 5")
        return
    ORDER_OPEN = mt5.ORDER_TYPE_SELL if IS_SHORT else mt5.ORDER_TYPE_BUY
    ORDER_CLOSE = mt5.ORDER_TYPE_BUY if IS_SHORT else mt5.ORDER_TYPE_SELL
    TARGET_ENTRADA = -1 if IS_SHORT else 1
    TARGET_SALIDA = 1 if IS_SHORT else -1
    last_processed_closed_time = None
    while True:
        try:
            metodo_ma, vela_min, lookback = verificar_y_optimizar_semana(SYMBOL)
        except Exception as e:
            print(f"[{ts()}] Error: {e}. Reintentando...")
            time.sleep(60)
            continue

        max_elementos = lookback + 100
        print(f"[{ts()}] Bot ejecutandose | {SYMBOL} ({DIRECTION}) | {metodo_ma} (TF: {vela_min}m, LB: {lookback})")
        velas_cache, last_m1_time = inicializar_cache_velas(SYMBOL, vela_min, max_elementos)

        while True:
            ahora = obtener_tiempo_servidor()

            if ahora.weekday() == 4 and ahora.hour == 23 and ahora.minute >= 50:

                if verificar_posicion_activa():
                    print(f"[{ts()}] Viernes detectado. Ejecutando Cierre_Viernes")
                    ejecutar_orden(ORDER_CLOSE, "Cierre Viernes")
                esperar_hasta_lunes()
                break

            esperar_hasta_siguiente_vela(vela_min)
            posicion_abierta = verificar_posicion_activa()
            print(posicion_abierta)
            last_m1_time = actualizar_cache_velas(SYMBOL, vela_min, candles_cache := velas_cache, last_m1_time, max_elementos)

            if len(velas_cache['time']) < lookback + 5:
                continue

            mid_arr = np.array(velas_cache['bid'])[:-1]
            time_arr = np.array(velas_cache['time'])[:-1]
            current_closed_time = time_arr[-1]

            if last_processed_closed_time == current_closed_time:
                continue

            last_processed_closed_time = current_closed_time
            señal = evaluar_señal_mr(mid_arr, metodo_ma, lookback)
            print(señal)
            tick_current = mt5.symbol_info_tick(SYMBOL)
            precio_ejecucion = tick_current.bid if tick_current else mid_arr[-1]

            if señal == TARGET_SALIDA and posicion_abierta:
                if ejecutar_orden(ORDER_CLOSE, "Cierre Senal"):
                    print(f"[{ts()}] Cierre por senal: {precio_ejecucion:.5f}")
            elif señal == TARGET_ENTRADA and not posicion_abierta:
                if ejecutar_orden(ORDER_OPEN, f"Entrada Directa {DIRECTION}"):
                    print(f"[{ts()}] Entrada ejecutada: {precio_ejecucion:.5f}")

if __name__ == "__main__":

    main()
