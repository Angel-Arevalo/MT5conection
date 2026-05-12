import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import talib
from collections import deque
import sys
import os

sys.path.append(os.path.abspath('./optimal-moving-average'))
import keys
from find_best import opti_main

SYMBOL          = "US500_SPOT"
MAGIC_NUMBER    = 1002
BROKER_OFFSET   = 3
N_ALCISTAS      = 2
N_CONFIRM       = 2
ATR_PERIODS     = 15
keys.calls      = 15
#keys.methods    = {"SMA", "EMA"}
#keys.candles    = 1
#keys.lookbacks  = 20

FAST_METHODS = {
    "SMA":      talib.SMA,
    "EMA":      talib.EMA,
    "WMA":      talib.WMA,
    "DEMA":     talib.DEMA,
    "TEMA":     talib.TEMA,
    "TRIMA":    talib.TRIMA,
    "KAMA":     talib.KAMA,
    "T3":       talib.T3,
    "MIDPOINT": talib.MIDPOINT,
}

def ts() -> str:
    return obtener_tiempo_servidor().strftime("%d/%m %H:%M:%S")

def obtener_tiempo_servidor() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=BROKER_OFFSET)

def obtener_filling_mode(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_FOK
    m = info.filling_mode
    if m & 1:
        return mt5.ORDER_FILLING_FOK
    if m & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def calcular_volumen(symbol: str, porcentaje: float = 0.40,
                     apalancamiento_deseado: int = 10) -> float:
    info   = mt5.symbol_info(symbol)
    cuenta = mt5.account_info()
    if info is None or cuenta is None:
        return 0.0
    capital         = cuenta.margin_free * porcentaje
    margin_1lot     = mt5.order_calc_margin(mt5.ORDER_TYPE_SELL, symbol, 1.0, info.bid)
    if not margin_1lot:
        return 0.0
    nocional        = margin_1lot * cuenta.leverage
    margin_estricto = nocional / apalancamiento_deseado
    lote = (capital / margin_estricto // info.volume_step) * info.volume_step
    return max(info.volume_min, min(lote, info.volume_max))


def ejecutar_orden(tipo_orden, comentario: str = ""):
    filling = obtener_filling_mode(SYMBOL)
    info    = mt5.symbol_info(SYMBOL)
    tick    = mt5.symbol_info_tick(SYMBOL)
    if info is None or tick is None:
        return None

    if tipo_orden == mt5.ORDER_TYPE_SELL:
        res = mt5.order_send({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       SYMBOL,
            "volume":       float(calcular_volumen(SYMBOL)),
            "type":         mt5.ORDER_TYPE_SELL,
            "price":        float(tick.bid),
            "magic":        int(MAGIC_NUMBER),
            "comment":      str(comentario),
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        })
        return res

    elif tipo_orden == mt5.ORDER_TYPE_BUY: 
        posiciones = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        if posiciones:
            for p in posiciones:
                mt5.order_send({
                    "action":       mt5.TRADE_ACTION_DEAL,
                    "symbol":       SYMBOL,
                    "volume":       float(p.volume),
                    "type":         mt5.ORDER_TYPE_BUY,
                    "position":     int(p.ticket),
                    "price":        float(mt5.symbol_info_tick(SYMBOL).ask),
                    "magic":        int(MAGIC_NUMBER),
                    "comment":      str(comentario),
                    "type_time":    mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                })
        return True

def obtener_data_optimizacion(symbol: str) -> pd.DataFrame | None:
    ahora = obtener_tiempo_servidor()
    lunes_actual = (ahora - timedelta(days=ahora.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0)
    viernes_fin  = (lunes_actual - timedelta(days=3)).replace(
        hour=23, minute=59, second=59)
    lunes_inicio = lunes_actual - timedelta(weeks=4)

    print(f"[{ts()}] Ventana optimización: "
          f"{lunes_inicio:%Y-%m-%d} → {viernes_fin:%Y-%m-%d}")

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1,
                                 lunes_inicio, viernes_fin)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    point = mt5.symbol_info(symbol).point

    df_final = pd.DataFrame({
        "time": df['time'],
        "bid":  df['close'],
        "ask":  df['close'] + (df['spread'] * point),
    })
    return df_final.set_index("time").dropna()

def esperar_hasta_lunes():
    ahora = obtener_tiempo_servidor()
    dias  = (7 - ahora.weekday()) % 7 or 7
    proximo_lunes = (ahora + timedelta(days=dias)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    secs = (proximo_lunes - ahora).total_seconds()
    print(f"[{ts()}] Durmiendo {secs / 3600:.2f}h hasta el lunes.")
    if secs > 0:
        time.sleep(secs)


def calcular_ma(close_arr: np.ndarray, metodo: str, lookback: int) -> np.ndarray:
    return FAST_METHODS[metodo](close_arr, timeperiod=lookback)

def ma_actual(close_arr: np.ndarray, metodo: str, lookback: int) -> float:
    ma     = calcular_ma(close_arr, metodo, lookback)
    valido = ma[~np.isnan(ma)]
    return float(valido[-1]) if len(valido) else 0.0

def calcular_spread(close_arr: np.ndarray, metodo: str,
                    lookback: int) -> np.ndarray:
    ma = calcular_ma(close_arr, metodo, lookback)
    return close_arr - ma

def obtener_señal(close_arr: np.ndarray, metodo: str, lookback: int) -> int:
    ma = calcular_ma(close_arr, metodo, lookback)

    if ma[-2] >= close_arr[-2] and ma[-1] < close_arr[-1]:
        return -1

    if ma[-2] < close_arr[-2] and ma[-1] >= close_arr[-1]:
        return 1

    return 0

def estimar_ou(spread: np.ndarray):
    S_prev = spread[:-1]
    S_curr = spread[1:]
    N = len(S_curr)
    if N < 3:
        return None, None, None

    Sx  = S_prev.sum()
    Sy  = S_curr.sum()
    Sxx = (S_prev ** 2).sum()
    Sxy = (S_prev * S_curr).sum()
    Syy = (S_curr ** 2).sum()

    denom = N * Sxx - Sx ** 2
    if abs(denom) < 1e-14:
        return None, None, None

    beta = (N * Sxy - Sx * Sy) / denom
    if beta <= 0 or beta >= 1:
        return None, None, None

    theta = -np.log(beta)
    sigma2_eps = (
        (Syy - 2 * beta * Sxy + beta ** 2 * Sxx) / N
        - (Sy / N - beta * Sx / N) ** 2
    )
    if sigma2_eps <= 1e-14:
        return None, None, None

    exp_m2t = np.exp(-2 * theta)
    den_sig = 1 - exp_m2t
    if den_sig < 1e-14:
        return None, None, None

    sigma2 = sigma2_eps * 2 * theta / den_sig
    sigma  = np.sqrt(sigma2)

    v = sigma2 / (2 * theta) * den_sig
    if v <= 1e-14:
        return None, None, None

    residuos = S_curr - S_prev * beta
    logL = (-N / 2 * np.log(2 * np.pi * v)
            - np.sum(residuos ** 2) / (2 * v))

    return theta, sigma, logL

def estimar_gbm(spread: np.ndarray):
    dS = np.diff(spread)
    N  = len(dS)
    if N < 3:
        return None, None, None

    mu    = dS.mean()
    sigma = dS.std(ddof=1)
    if sigma <= 1e-14:
        return None, None, None

    logL = (-N / 2 * np.log(2 * np.pi * sigma ** 2)
            - np.sum((dS - mu) ** 2) / (2 * sigma ** 2))
    return mu, sigma, logL

def detectar_regimen(close_arr: np.ndarray, metodo: str, lookback: int) -> str:

    spread = calcular_spread(close_arr, metodo, lookback)
    valido = spread[~np.isnan(spread)]

    if len(valido) < lookback + 1:
        return "INDETERMINADO"

    ventana = valido[-(lookback + 1):]

    _, _, logL_OU        = estimar_ou(ventana)
    mu_gbm, _, logL_GBM  = estimar_gbm(ventana)

    if any(v is None for v in [logL_OU, mu_gbm, logL_GBM]):
        return "INDETERMINADO"

    LR = logL_OU - logL_GBM

    if LR > 0:
        return "REVERSION"
    if mu_gbm > 0:
        return "ALCISTA"
    return "BAJISTA_GBM"

def main():
    if not mt5.initialize():
        print("Error al inicializar MT5")
        return

    while True:

        print(f"[{ts()}] Optimizando MA para CORTOS (4 semanas sin semana actual)...")
        data_opt = obtener_data_optimizacion(SYMBOL)
        if data_opt is None:
            print(f"[{ts()}] Sin datos de optimización. Reintentando en 60s.")
            time.sleep(60)
            continue

        params_ma  = opti_main(data_opt, is_bid=True, shorts=True)
        metodo_ma  = params_ma[0]
        vela_min   = int(params_ma[1])
        lookback   = int(params_ma[2])
        vela_seg   = vela_min * 60
        max_espera = max(lookback // 2, N_ALCISTAS + N_CONFIRM + 1)

        print(f"[{ts()}] MA ganadora (cortos) → método={metodo_ma}  "
              f"vela={vela_min}min  lookback={lookback}  "
              f"max_espera={max_espera} velas")

        minutos_buf   = max(lookback + 5, ATR_PERIODS + 1) * vela_min
        initial_rates = mt5.copy_rates_from_pos(
            SYMBOL, mt5.TIMEFRAME_M1, 0, minutos_buf)

        def _muestra(arr, paso):
            return arr[::-1][::paso][::-1].astype(np.float64)

        close_dq  = deque(_muestra(initial_rates['close'],  vela_min), maxlen=minutos_buf)
        high_dq   = deque(_muestra(initial_rates['high'],   vela_min), maxlen=minutos_buf)
        low_dq    = deque(_muestra(initial_rates['low'],    vela_min), maxlen=minutos_buf)
        spread_dq = deque(_muestra(initial_rates['spread'], vela_min), maxlen=minutos_buf)

        proxima_vela_ts = (
            (obtener_tiempo_servidor().timestamp() // vela_seg) + 1
        ) * vela_seg

        vendido                = False
        en_señal               = False
        velas_alcistas_consec  = 0
        velas_reversion_consec = 0
        velas_espera           = 0
        alcista_confirmado     = False
        ma_compra_obj          = 0.0

        while True:
            ahora    = obtener_tiempo_servidor()
            ahora_ts = ahora.timestamp()

            hora_pre_viernes  = None
            es_zona_prohibida = False

            if ahora.weekday() == 4:
                hora_limite      = ahora.replace(hour=23, minute=57,
                                                 second=0, microsecond=0)
                hora_pre_viernes = hora_limite - timedelta(minutes=vela_min)
                hora_bloqueo     = hora_limite - timedelta(minutes=2 * vela_min)
                es_zona_prohibida = ahora >= hora_bloqueo

                if ahora >= hora_pre_viernes:
                    posiciones = mt5.positions_get(symbol=SYMBOL,
                                                   magic=MAGIC_NUMBER)
                    if posiciones:
                        secs = (hora_limite - ahora).total_seconds()
                        if secs > 0:
                            time.sleep(secs)
                        precio_entrada = posiciones[0].price_open
                        objetivo       = precio_entrada * (1 - 0.00004)  # target bajista
                        while True:
                            t = obtener_tiempo_servidor()
                            if t.hour == 23 and t.minute >= 59:
                                ejecutar_orden(mt5.ORDER_TYPE_BUY,
                                               "Emergencia Viernes")
                                break
                            tick = mt5.symbol_info_tick(SYMBOL)
                            if tick and tick.ask <= objetivo:
                                ejecutar_orden(mt5.ORDER_TYPE_BUY,
                                               "Cierre Inteligente Viernes")
                                break
                            time.sleep(0.1)

                    vendido  = en_señal = False
                    esperar_hasta_lunes()
                    break

            if ahora_ts >= proxima_vela_ts:
                velas_perdidas   = int((ahora_ts - proxima_vela_ts)
                                       // vela_seg) + 1
                proxima_vela_ts += velas_perdidas * vela_seg

                new_rates = mt5.copy_rates_from_pos(
                    SYMBOL, mt5.TIMEFRAME_M1, 0, velas_perdidas * vela_min)
                if new_rates is not None and len(new_rates) > 0:
                    close_dq.extend(_muestra(new_rates['close'],  vela_min))
                    high_dq.extend(_muestra(new_rates['high'],    vela_min))
                    low_dq.extend(_muestra(new_rates['low'],      vela_min))
                    spread_dq.extend(_muestra(new_rates['spread'],vela_min))

                close_arr = np.array(close_dq)

                ma_val = ma_actual(close_arr, metodo_ma, lookback)
                if vendido:
                    ma_compra_obj = ma_val

                señal = obtener_señal(close_arr, metodo_ma, lookback)

                if señal == 1 and vendido:
                    ejecutar_orden(mt5.ORDER_TYPE_BUY, "Cierre corto vela")
                    vendido   = False
                    en_señal  = False
                    velas_alcistas_consec = velas_reversion_consec = 0
                    velas_espera          = 0
                    alcista_confirmado    = False
                    print(f"[{ts()}] CIERRE CORTO en vela (respaldo) — "
                          f"MA={ma_val:.5f}")

                elif señal == 1 and not vendido:

                    en_señal              = False
                    velas_alcistas_consec = velas_reversion_consec = 0
                    velas_espera          = 0
                    alcista_confirmado    = False

                elif (señal == -1 and not vendido
                      and not en_señal and not es_zona_prohibida):
                    en_señal              = True
                    velas_alcistas_consec = 0
                    velas_reversion_consec= 0
                    velas_espera          = 0
                    alcista_confirmado    = False
                    print(f"[{ts()}] Señal de corto — "
                          f"precio={close_arr[-1]:.5f}  MA={ma_val:.5f} — "
                          f"iniciando monitoreo OU/GBM")

                if en_señal and not vendido:
                    regimen = detectar_regimen(close_arr, metodo_ma, lookback)
                    velas_espera += 1

                    if regimen == "ALCISTA":

                        velas_alcistas_consec  += 1
                        velas_reversion_consec  = 0
                        if velas_alcistas_consec >= N_ALCISTAS:
                            alcista_confirmado = True
                        print(f"[{ts()}] Régimen ALCISTA "
                              f"({velas_alcistas_consec} velas) — "
                              f"esperando agotamiento")

                    else:
                        velas_alcistas_consec = 0

                        if alcista_confirmado:

                            velas_reversion_consec += 1
                            print(f"[{ts()}] Régimen {regimen} "
                                  f"({velas_reversion_consec}/{N_CONFIRM} "
                                  f"para confirmar corto)")

                            if velas_reversion_consec >= N_CONFIRM:
                                res = ejecutar_orden(mt5.ORDER_TYPE_SELL,
                                                     "Entrada corto OU/GBM")
                                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                    vendido               = True
                                    en_señal              = False
                                    ma_compra_obj         = ma_val
                                    print(f"[{ts()}] CORTO ejecutado tras "
                                          f"agotamiento alcista — "
                                          f"precio={close_arr[-1]:.5f}  "
                                          f"MA={ma_val:.5f}")
                        else:
                            # Sin impulso alcista previo → corto directo
                            res = ejecutar_orden(mt5.ORDER_TYPE_SELL,
                                                 "Entrada corto directa")
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                vendido       = True
                                en_señal      = False
                                ma_compra_obj = ma_val
                                print(f"[{ts()}] CORTO directo "
                                      f"(régimen no alcista) — "
                                      f"precio={close_arr[-1]:.5f}  "
                                      f"MA={ma_val:.5f}")


                    if en_señal and velas_espera >= max_espera:
                        en_señal              = False
                        velas_alcistas_consec = 0
                        velas_reversion_consec= 0
                        alcista_confirmado    = False
                        print(f"[{ts()}] Señal cancelada — "
                              f"máximo de espera alcanzado "
                              f"({max_espera} velas)")

            t_espera = max(0.1, proxima_vela_ts - obtener_tiempo_servidor().timestamp())
            if ahora.weekday() == 4 and hora_pre_viernes is not None:
                t_espera = min(
                    t_espera,
                    max(0.1, (hora_pre_viernes - ahora).total_seconds()),
                )
                print(f"[{ts()}] Durmiendo {t_espera:.1f}s")
                time.sleep(max(0.1, t_espera))

            print(f"{ts()}: durmiendo {t_espera:.2f}")
            time.sleep(t_espera)


if __name__ == "__main__":
    main()
