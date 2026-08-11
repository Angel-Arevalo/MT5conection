import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

sys.path.append(os.path.abspath('../optimal-moving-average'))

import keys
from find_best import opti_main

SYMBOL           = "AUDCAD_"
keys.calls       = 25

CAPITAL_LONG     = 10_000.0
CAPITAL_SHORT    = 10_000.0
APALANCAMIENTO   = 2

YEARS            = [2026]

COMISION_FALLBACK = 0.0
RISK_FREE_RATE    = 0.0

SWAP_BY_POINTS   = 0
SWAP_BY_MONEY    = 1
SWAP_BY_PCT_OPEN = 2
SWAP_BY_PCT_CUR  = 3

PRE_CALC_DIR = "pre_calc"

FAST_METHODS = {
    "SMA": talib.SMA, "EMA": talib.EMA, "WMA": talib.WMA,
    "DEMA": talib.DEMA, "TEMA": talib.TEMA, "TRIMA": talib.TRIMA,
    "KAMA": talib.KAMA, "T3": talib.T3, "MIDPOINT": talib.MIDPOINT,
}


def _ruta_pre_calc(symbol: str) -> str:
    return os.path.join(PRE_CALC_DIR, f"{symbol}.csv")


def _cargar_pre_calc(symbol: str) -> dict:
    ruta = _ruta_pre_calc(symbol)
    if not os.path.exists(ruta):
        return {}
    try:
        df = pd.read_csv(ruta, dtype={'semana': str, 'met_l': str, 'met_s': str})
        cache = {}
        for _, row in df.iterrows():
            cache[row['semana']] = {
                'met_l':  row['met_l'],
                'vela_l': int(row['vela_l']),
                'lb_l':   int(row['lb_l']),
                'met_s':  row['met_s'],
                'vela_s': int(row['vela_s']),
                'lb_s':   int(row['lb_s']),
            }
        print(f"  [pre_calc] {len(cache)} semanas cargadas de '{ruta}'")
        return cache
    except Exception as e:
        print(f"  [pre_calc] Error leyendo caché '{ruta}': {e}")
        return {}


def _guardar_pre_calc(symbol: str, cache: dict) -> None:
    os.makedirs(PRE_CALC_DIR, exist_ok=True)
    ruta = _ruta_pre_calc(symbol)
    filas = [
        {
            'semana': semana,
            'met_l':  v['met_l'],  'vela_l': v['vela_l'],  'lb_l': v['lb_l'],
            'met_s':  v['met_s'],  'vela_s': v['vela_s'],  'lb_s': v['lb_s'],
        }
        for semana, v in sorted(cache.items())
    ]
    pd.DataFrame(filas).to_csv(ruta, index=False)


def _ma(arr, metodo, lb):
    return FAST_METHODS[metodo](arr, timeperiod=lb)


def evaluar_señal_mr(arr, metodo, lb):
    ma = _ma(arr, metodo, lb)
    if len(ma) < 2 or np.isnan(ma[-1]) or np.isnan(ma[-2]):
        return 0
    if arr[-2] >= ma[-2] and arr[-1] < ma[-1]:
        return 1
    if arr[-2] <= ma[-2] and arr[-1] > ma[-1]:
        return -1
    return 0


def calcular_lotes_y_pnl(precio_entrada, pnl_price, capital, contract_size):
    valor_lote = precio_entrada * contract_size
    if valor_lote <= 0:
        return 0.0, 0.0
    lotes     = (capital * APALANCAMIENTO) / valor_lote
    pnl_bruto = pnl_price * contract_size * lotes
    return lotes, pnl_bruto


def calcular_swap(f_entrada, f_salida, lotes, swap_rate, swap_mode,
                  precio_entrada, contract_size, tick_value, tick_size,
                  rollover3days):
    if f_salida <= f_entrada or swap_rate == 0.0:
        return 0.0
    cur   = f_entrada.replace(hour=0, minute=0, second=0, microsecond=0)
    fin   = f_salida.replace(hour=0, minute=0, second=0, microsecond=0)
    total = 0.0
    while cur < fin:
        mult = 3 if cur.weekday() == rollover3days else 1
        if swap_mode == SWAP_BY_POINTS:
            vpp   = (tick_value / tick_size) if tick_size > 0 else 1.0
            noche = swap_rate * vpp * lotes * mult
        elif swap_mode in (SWAP_BY_PCT_OPEN, SWAP_BY_PCT_CUR):
            valor_lote = precio_entrada * contract_size
            noche = (swap_rate / 100.0 / 365.0) * valor_lote * lotes * mult
        else:
            noche = swap_rate * lotes * mult
        total += noche
        cur   += timedelta(days=1)
    return total


def obtener_comision_rt(symbol, year):
    deals = mt5.history_deals_get(datetime(year-1, 1, 1), datetime(year+1, 1, 1))
    if deals is None or len(deals) == 0:
        return COMISION_FALLBACK
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df = df[(df['symbol'] == symbol) & (df['commission'] != 0) & (df['volume'] > 0)].copy()
    if df.empty:
        return COMISION_FALLBACK
    return (df['commission'].abs() / df['volume']).mean() * 2


def _semana_lado(df_semana, lunes_actual, viernes_actual, metodo, lb, vela_min, es_long=True):
    trades = []
    if df_semana.empty:
        return trades

    vela_seg = vela_min * 60
    # Forzar resolución a segundos de forma segura sin importar la versión de Pandas
    ts_sec = df_semana.index.astype('datetime64[s]').astype('int64')
    custom_ts = (ts_sec // vela_seg) * vela_seg

    df_temp = df_semana.copy()
    df_temp['custom_open'] = pd.to_datetime(custom_ts, unit='s')

    df_custom = df_temp.groupby('custom_open').agg({
        'bid': ['first', 'last'],
        'ask': ['first', 'last'],
        'high': 'max',
        'low': 'min'
    })
    df_custom.columns = ['bid_open', 'bid_close', 'ask_open', 'ask_close', 'high', 'low']

    custom_times = df_custom.index
    start_indices = np.where(custom_times >= lunes_actual)[0]
    if len(start_indices) == 0:
        return trades

    start_idx = start_indices[0]
    posicionado = False
    posicion = None

    target_entrada = 1 if es_long else -1
    target_salida = -1 if es_long else 1

    for i in range(start_idx, len(df_custom)):
        t = custom_times[i]

        bid_open = float(df_custom['bid_open'].iloc[i])
        ask_open = float(df_custom['ask_open'].iloc[i])
        high_curr = float(df_custom['high'].iloc[i])
        low_curr = float(df_custom['low'].iloc[i])

        if t >= viernes_actual or (t.weekday() == 4 and t.hour == 23 and t.minute >= 50):
            if posicionado:
                precio_salida = bid_open if es_long else ask_open
                if es_long:
                    mae_p = posicion['precio_in'] - min(posicion['min_low'], low_curr)
                    mfe_p = max(posicion['max_high'], high_curr) - posicion['precio_in']
                    pnl_p = precio_salida - posicion['precio_in']
                else:
                    mae_p = max(posicion['max_high'], high_curr) - posicion['precio_in']
                    mfe_p = posicion['precio_in'] - min(posicion['min_low'], low_curr)
                    pnl_p = posicion['precio_in'] - precio_salida

                trades.append({
                    'fecha_entrada': posicion['fecha'],
                    'fecha_salida': t,
                    'precio_entrada': posicion['precio_in'],
                    'precio_salida': precio_salida,
                    'pnl_price': pnl_p,
                    'tipo': 'Cierre_Viernes',
                    'mae_price': max(0.0, mae_p),
                    'mfe_price': max(0.0, mfe_p),
                })
                posicionado = False
                posicion = None
            break

        bid_arr = df_custom['bid_close'].iloc[:i].values

        if len(bid_arr) < lb + 5:
            continue

        s = evaluar_señal_mr(bid_arr, metodo, lb)

        if posicionado:
            posicion['max_high'] = max(posicion['max_high'], high_curr)
            posicion['min_low']  = min(posicion['min_low'],  low_curr)

            if s == target_salida:
                precio_salida = bid_open if es_long else ask_open
                if es_long:
                    mae_p = posicion['precio_in'] - posicion['min_low']
                    mfe_p = posicion['max_high']  - posicion['precio_in']
                    pnl_p = precio_salida - posicion['precio_in']
                else:
                    mae_p = posicion['max_high']  - posicion['precio_in']
                    mfe_p = posicion['precio_in'] - posicion['min_low']
                    pnl_p = posicion['precio_in'] - precio_salida

                trades.append({
                    'fecha_entrada': posicion['fecha'],
                    'fecha_salida': t,
                    'precio_entrada': posicion['precio_in'],
                    'precio_salida': precio_salida,
                    'pnl_price': pnl_p,
                    'tipo': 'Cierre_Señal',
                    'mae_price': max(0.0, mae_p),
                    'mfe_price': max(0.0, mfe_p),
                })
                posicionado = False
                posicion = None

        elif not posicionado:
            if s == target_entrada:
                precio_in = ask_open if es_long else bid_open
                posicion = {
                    'precio_in': precio_in,
                    'fecha':     t,
                    'max_high':  high_curr,
                    'min_low':   low_curr,
                }
                posicionado = True

    return trades

def _contabilizar(trades, capital_inicial, swap_rate, swap_mode,
                  contract_size, tick_value, tick_size, rollover3days, comision_rt):
    df = pd.DataFrame(trades)
    if df.empty:
        return df, capital_inicial, 0.0

    bal = capital_inicial
    res = 0.0
    pnl_bruto_l = []; pnl_neto_l = []; bal_l = []
    res_l = []; pat_l = []; lot_l = []
    swp_l = []; com_l = []; ret_l = []
    mae_usd_l = []; mfe_usd_l = []

    for _, row in df.iterrows():
        lotes, pnl_bruto = calcular_lotes_y_pnl(
            row['precio_entrada'], row['pnl_price'], bal, contract_size)
        com  = comision_rt * lotes
        swap = calcular_swap(
            row['fecha_entrada'], row['fecha_salida'],
            lotes, swap_rate, swap_mode,
            row['precio_entrada'], contract_size,
            tick_value, tick_size, rollover3days)
        pnl_neto = pnl_bruto - com + swap
        mae_usd  = row.get('mae_price', 0.0) * contract_size * lotes
        mfe_usd  = row.get('mfe_price', 0.0) * contract_size * lotes

        prev = bal
        bal += pnl_neto
        if bal > capital_inicial:
            res += bal - capital_inicial
            bal  = capital_inicial
        elif bal < capital_inicial and res > 0:
            t    = min(capital_inicial - bal, res)
            res -= t; bal += t

        pat = bal + res
        ret = pnl_neto / prev

        pnl_bruto_l.append(pnl_bruto); pnl_neto_l.append(pnl_neto)
        bal_l.append(bal); res_l.append(res); pat_l.append(pat)
        lot_l.append(lotes); swp_l.append(swap); com_l.append(com); ret_l.append(ret)
        mae_usd_l.append(mae_usd); mfe_usd_l.append(mfe_usd)

    df['lotes']               = lot_l
    df['swap_usd']            = swp_l
    df['comision_usd']        = com_l
    df['pnl_bruto_usd']       = pnl_bruto_l
    df['pnl_neto_usd']        = pnl_neto_l
    df['mae_usd']             = mae_usd_l
    df['mfe_usd']             = mfe_usd_l
    df['retorno_trade']       = ret_l
    df['balance_operativo']   = bal_l
    df['ganancias_apartadas'] = res_l
    df['patrimonio_total']    = pat_l

    df['peak']         = np.maximum(df['patrimonio_total'].cummax(), capital_inicial)
    df['drawdown_abs'] = df['peak'] - df['patrimonio_total']
    df['drawdown_pct'] = df['drawdown_abs'] / df['peak'] * 100

    return df, bal, res


def imprimir_tabla(lista_dfs, lado: str):
    dfs_validos = [df for df in lista_dfs if isinstance(df, pd.DataFrame) and not df.empty]
    if not dfs_validos:
        print(f"\n[{lado}] Sin operaciones registradas.")
        return

    df = pd.concat(dfs_validos, ignore_index=True)

    total_trades = len(df)
    wins         = df[df['pnl_neto_usd'] > 0]
    losses       = df[df['pnl_neto_usd'] <= 0]
    num_wins     = len(wins)
    num_losses   = len(losses)
    win_rate     = (num_wins / total_trades * 100) if total_trades > 0 else 0.0

    pnl_bruto    = df['pnl_bruto_usd'].sum()
    comisiones   = df['comision_usd'].sum()
    swaps        = df['swap_usd'].sum()
    pnl_neto     = df['pnl_neto_usd'].sum()

    gross_profit  = wins['pnl_neto_usd'].sum()
    gross_loss    = abs(losses['pnl_neto_usd'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    max_dd_pct    = df['drawdown_pct'].max() if 'drawdown_pct' in df.columns else 0.0

    print(f"\n{'='*65}")
    print(f" RESUMEN DE RESULTADOS — {lado}")
    print(f"{'='*65}")
    print(f"  Total Trades        : {total_trades}")
    print(f"  Trades Ganadores    : {num_wins} ({win_rate:.2f}%)")
    print(f"  Trades Perdedores   : {num_losses}")
    print(f"  -------------------------------------------")
    print(f"  PnL Bruto (USD)     : ${pnl_bruto:,.2f}")
    print(f"  Comisiones (USD)    : ${comisiones:,.2f}")
    print(f"  Swaps (USD)         : ${swaps:,.2f}")
    print(f"  -------------------------------------------")
    print(f"  PnL Neto Total (USD): ${pnl_neto:,.2f}")
    print(f"  Profit Factor       : {profit_factor:.2f}")
    print(f"  Max Drawdown (%)    : {max_dd_pct:.2f}%")
    print(f"{'='*65}\n")


def backtest_año(year: int):
    if not mt5.initialize():
        print(f"[{year}] Error inicializando MT5"); return None, None

    if not mt5.symbol_select(SYMBOL, True):
        print(f"[{year}] No se pudo seleccionar {SYMBOL}")
        mt5.shutdown(); return None, None

    info = mt5.symbol_info(SYMBOL)
    if info is None:
        mt5.shutdown(); return None, None

    point         = info.point
    tick_size     = info.trade_tick_size
    tick_value    = info.trade_tick_value
    contract_size = info.trade_contract_size
    swap_mode     = info.swap_mode
    swap_long_r   = info.swap_long
    swap_short_r  = info.swap_short
    rollover3days = info.swap_rollover3days

    comision_rt = obtener_comision_rt(SYMBOL, year)

    print(f"\n  Info instrumento: contract_size={contract_size}, "
          f"tick_size={tick_size}, tick_value={tick_value}, swap_mode={swap_mode}")

    rates = mt5.copy_rates_range(
        SYMBOL, mt5.TIMEFRAME_M1,
        datetime(year-1, 11, 1), datetime(year, 12, 28)
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"[{year}] Sin datos"); return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df['bid'] = df['close'].astype(float)
    df['ask'] = (df['close'] + df['spread'] * point).astype(float)

    trades_long  = []
    trades_short = []
    lunes_rango  = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='W-MON')
    ahora        = datetime.now()

    cache_pre = _cargar_pre_calc(SYMBOL)

    for lunes in lunes_rango:
        viernes_cierre = (lunes + timedelta(days=4)).replace(
            hour=23, minute=59, second=59)
        if viernes_cierre >= ahora:
            continue

        semana_key = str(lunes.date())
        viernes    = (lunes + timedelta(days=4)).replace(hour=23, minute=50, second=0)

        if semana_key in cache_pre:
            vals  = cache_pre[semana_key]
            met_l, vela_l, lb_l = vals['met_l'], vals['vela_l'], vals['lb_l']
            met_s, vela_s, lb_s = vals['met_s'], vals['vela_s'], vals['lb_s']
            print(f"  {lunes.date()} | [CACHE] "
                  f"Long: {met_l} lb={lb_l} v={vela_l} | "
                  f"Short: {met_s} lb={lb_s} v={vela_s}")
        else:
            l_ini  = lunes - timedelta(weeks=4)
            v_fin  = (lunes - timedelta(days=3)).replace(
                hour=23, minute=59, second=59)
            df_opt = df.loc[l_ini:v_fin].copy()

            if df_opt.empty:
                continue
            if (df_opt.index[-1] - df_opt.index[0]).days < 20:
                print(f"  {lunes.date()} | [SKIP] datos insuficientes para optimización")
                continue

            inp = df_opt[['bid', 'ask']].copy()
            for c in ['open', 'high', 'low', 'close']:
                if c in df_opt.columns:
                    inp[c] = df_opt[c]

            p_l = opti_main(inp, is_bid=True, verbose=True, shorts=False)
            p_s = opti_main(inp, is_bid=True, verbose=True, shorts=True)

            met_l, vela_l, lb_l = p_l[0], int(p_l[1]), int(p_l[2])
            met_s, vela_s, lb_s = p_s[0], int(p_s[1]), int(p_s[2])

            cache_pre[semana_key] = {
                'met_l':  met_l, 'vela_l': vela_l, 'lb_l': lb_l,
                'met_s':  met_s, 'vela_s': vela_s, 'lb_s': lb_s,
            }
            _guardar_pre_calc(SYMBOL, cache_pre)

            print(f"  {lunes.date()} | [NUEVO] "
                  f"Long: {met_l} lb={lb_l} v={vela_l} | "
                  f"Short: {met_s} lb={lb_s} v={vela_s}")

        dias_hist_l = max(21, int(((lb_l + 30) * vela_l) / 1440 * 2.5))
        dias_hist_s = max(21, int(((lb_s + 30) * vela_s) / 1440 * 2.5))

        buf_l = lunes - timedelta(days=dias_hist_l)
        buf_s = lunes - timedelta(days=dias_hist_s)

        sem_l = df.loc[buf_l:viernes]
        sem_s = df.loc[buf_s:viernes]

        if not sem_l.empty:
            trades_long  += _semana_lado(
                sem_l, lunes, viernes, met_l, lb_l, vela_l, es_long=True)
        if not sem_s.empty:
            trades_short += _semana_lado(
                sem_s, lunes, viernes, met_s, lb_s, vela_s, es_long=False)

    df_long, _, _ = _contabilizar(
        trades_long, CAPITAL_LONG, swap_long_r, swap_mode,
        contract_size, tick_value, tick_size, rollover3days, comision_rt)

    df_short, _, _ = _contabilizar(
        trades_short, CAPITAL_SHORT, swap_short_r, swap_mode,
        contract_size, tick_value, tick_size, rollover3days, comision_rt)

    return df_long, df_short

def filtrar_trades_por_fecha(lista_dfs, fecha_inicio: str, fecha_fin: str, lado: str = "LONG"):
    """
    Filtra e imprime en pantalla los trades ejecutados en un rango de fechas.
    Formato de fechas: 'YYYY-MM-DD'
    """
    dfs_validos = [df for df in lista_dfs if isinstance(df, pd.DataFrame) and not df.empty]
    if not dfs_validos:
        print(f"\n[{lado}] No hay datos para filtrar.")
        return pd.DataFrame()

    df_total = pd.concat(dfs_validos, ignore_index=True)

    f_ini = pd.to_datetime(fecha_inicio)
    f_fin = pd.to_datetime(fecha_fin).replace(hour=23, minute=59, second=59)

    mask = (df_total['fecha_entrada'] >= f_ini) & (df_total['fecha_entrada'] <= f_fin)
    df_filtrado = df_total.loc[mask].copy()

    if df_filtrado.empty:
        print(f"\n[{lado}] No se encontraron trades entre {fecha_inicio} y {fecha_fin}.")
        return df_filtrado

    cols_ver = [
        'fecha_entrada', 'fecha_salida', 'tipo', 
        'precio_entrada', 'precio_salida', 'lotes', 
        'pnl_bruto_usd', 'swap_usd', 'pnl_neto_usd'
    ]
    cols_existentes = [c for c in cols_ver if c in df_filtrado.columns]

    print(f"\n{'='*105}")
    print(f" DETALLE DE TRADES {lado} — ({fecha_inicio} a {fecha_fin}) | Total: {len(df_filtrado)}")
    print(f"{'='*105}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df_filtrado[cols_existentes].to_string(index=False))
    print(f"{'='*105}\n")

    return df_filtrado

if __name__ == "__main__":
    todos_long  = []
    todos_short = []

    for year in YEARS:
        print(f"\n{'#'*65}")
        print(f"  Backtest MR Directo {year}  —  {SYMBOL}")
        print(f"{'#'*65}")
        df_l, df_s = backtest_año(year)
        if df_l is not None and not df_l.empty:
            todos_long.append(df_l)
        if df_s is not None and not df_s.empty:
            todos_short.append(df_s)

    imprimir_tabla(todos_long,  "LONG")
    imprimir_tabla(todos_short, "SHORT")

    fecha_desde = "2026-07-28"
    fecha_hasta = "2026-07-31"

    filtrar_trades_por_fecha(todos_long,  fecha_desde, fecha_hasta, lado="LONG")
    filtrar_trades_por_fecha(todos_short, fecha_desde, fecha_hasta, lado="SHORT")
