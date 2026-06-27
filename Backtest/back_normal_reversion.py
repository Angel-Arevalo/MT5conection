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

SYMBOL           = "US100_Spot"
keys.calls       = 25

CAPITAL_LONG     = 10_000.0
CAPITAL_SHORT    = 10_000.0
APALANCAMIENTO   = 2

YEARS            = [2024, 2025, 2026]

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


def _generar_equity_chart(df_l, df_s, year, symbol):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#161625')

    ts_l = ts_s = None

    if df_l is not None and not df_l.empty:
        t0 = df_l['fecha_entrada'].iloc[0]

        pts_l = pd.concat([
            pd.Series([CAPITAL_LONG], index=[t0]),
            df_l.set_index('fecha_salida')['patrimonio_total']
        ]).sort_index()

        ts_l = pts_l

        pat_fin_l = pts_l.iloc[-1]
        ret_l = (pat_fin_l / CAPITAL_LONG - 1) * 100

        ax.plot(
            pts_l.index,
            pts_l.values,
            color='#2ecc71',
            lw=1.8,
            zorder=3,
            label=f'Long  {ret_l:+.1f}%  →  ${pat_fin_l:,.0f}'
        )

        ax.fill_between(
            pts_l.index,
            CAPITAL_LONG,
            pts_l.values,
            where=pts_l.values >= CAPITAL_LONG,
            color='#2ecc71',
            alpha=0.13
        )

        ax.fill_between(
            pts_l.index,
            CAPITAL_LONG,
            pts_l.values,
            where=pts_l.values < CAPITAL_LONG,
            color='#e74c3c',
            alpha=0.13
        )

    if df_s is not None and not df_s.empty:
        t0 = df_s['fecha_entrada'].iloc[0]

        pts_s = pd.concat([
            pd.Series([CAPITAL_SHORT], index=[t0]),
            df_s.set_index('fecha_salida')['patrimonio_total']
        ]).sort_index()

        ts_s = pts_s

        pat_fin_s = pts_s.iloc[-1]
        ret_s = (pat_fin_s / CAPITAL_SHORT - 1) * 100

        ax.plot(
            pts_s.index,
            pts_s.values,
            color='#e67e22',
            lw=1.8,
            zorder=3,
            label=f'Short  {ret_s:+.1f}%  →  ${pat_fin_s:,.0f}'
        )

        ax.fill_between(
            pts_s.index,
            CAPITAL_SHORT,
            pts_s.values,
            where=pts_s.values >= CAPITAL_SHORT,
            color='#e67e22',
            alpha=0.13
        )

        ax.fill_between(
            pts_s.index,
            CAPITAL_SHORT,
            pts_s.values,
            where=pts_s.values < CAPITAL_SHORT,
            color='#e74c3c',
            alpha=0.13
        )

    if ts_l is not None and ts_s is not None:

        idx = ts_l.index.union(ts_s.index).sort_values()

        eq_l = ts_l.reindex(idx).ffill().bfill()
        eq_s = ts_s.reindex(idx).ffill().bfill()

        ts_tot = eq_l + eq_s - CAPITAL_LONG

        cap_tot = CAPITAL_LONG

        pat_fin_tot = ts_tot.iloc[-1]
        ret_tot = (pat_fin_tot / cap_tot - 1) * 100

        ax.plot(
            ts_tot.index,
            ts_tot.values,
            color='#5dade2',
            lw=2.5,
            ls='--',
            zorder=5,
            label=f'Total Real  {ret_tot:+.1f}%  →  ${pat_fin_tot:,.0f}'
        )

        ax.axhline(
            cap_tot,
            color='#7f8c8d',
            ls=':',
            lw=1,
            alpha=0.55,
            label=f'Capital Real  ${cap_tot:,.0f}'
        )

        ax.fill_between(
            ts_tot.index,
            cap_tot,
            ts_tot.values,
            where=ts_tot.values >= cap_tot,
            color='#3498db',
            alpha=0.08
        )

        ax.fill_between(
            ts_tot.index,
            cap_tot,
            ts_tot.values,
            where=ts_tot.values < cap_tot,
            color='#e74c3c',
            alpha=0.08
        )

    elif ts_l is not None:

        ax.axhline(
            CAPITAL_LONG,
            color='#7f8c8d',
            ls=':',
            lw=1,
            alpha=0.55
        )

    elif ts_s is not None:

        ax.axhline(
            CAPITAL_SHORT,
            color='#7f8c8d',
            ls=':',
            lw=1,
            alpha=0.55
        )

    col_txt = '#c8d0d8'

    ax.set_title(
        f'{symbol}  ·  Equity MR Directo {year}',
        fontsize=14,
        fontweight='bold',
        color='white',
        pad=14
    )

    ax.set_ylabel(
        'Patrimonio ($)',
        color=col_txt,
        fontsize=10
    )

    ax.tick_params(
        colors=col_txt,
        labelsize=9
    )

    for spine in ax.spines.values():
        spine.set_color('#2c3e50')

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%b')
    )

    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
    )

    ax.grid(
        True,
        color='#1e2030',
        linewidth=0.8
    )

    ax.legend(
        facecolor='#0f0f1a',
        edgecolor='#2c3e50',
        labelcolor=col_txt,
        fontsize=9,
        loc='upper left'
    )

    fig.autofmt_xdate(
        rotation=0,
        ha='center'
    )

    fig.tight_layout(pad=1.5)

    ruta = os.path.expanduser(
        f"~/{symbol}_{year}_mr_directo_equity.png"
    )

    fig.savefig(
        ruta,
        dpi=150,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print(f"  Equity chart: {ruta}")

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


def calcular_sharpe(rets, rf=0.0):
    r = np.asarray(rets)
    if len(r) < 2:
        return 0.0
    ex  = r - rf
    std = ex.std(ddof=1)
    return 0.0 if std <= 1e-14 else np.sqrt(len(ex)) * ex.mean() / std


def calcular_rachas(pnl):
    mg = mp = rg = rp = 0
    for v in pnl:
        if v > 0:
            rg += 1; rp = 0; mg = max(mg, rg)
        else:
            rp += 1; rg = 0; mp = max(mp, rp)
    return mg, mp


def _semana_lado(df_semana, lunes_actual, viernes_actual, metodo, lb, vela_min, es_long=True):
    trades      = []
    posicionado = False
    posicion    = None

    try:
        start = df_semana.index.get_loc(df_semana.loc[lunes_actual:].index[0])
    except IndexError:
        return trades

    for i in range(start, len(df_semana), vela_min):
        w = df_semana.iloc[:i+1].iloc[::-1][::vela_min].iloc[::-1]
        if len(w) < lb + 5:
            continue

        arr  = w['bid'].values
        ask  = w['ask'].values
        high = w['high'].values
        low  = w['low'].values
        bid_ = float(arr[-1])
        ask_ = float(ask[-1])
        t    = w.index[-1]

        if posicionado:
            posicion['max_high'] = max(posicion['max_high'], float(high[-1]))
            posicion['min_low']  = min(posicion['min_low'],  float(low[-1]))

        if t >= viernes_actual:
            if posicionado:
                if es_long:
                    mae_p = posicion['precio_in'] - posicion['min_low']
                    mfe_p = posicion['max_high']  - posicion['precio_in']
                    pnl_p = bid_ - posicion['precio_in']
                else:
                    mae_p = posicion['max_high']  - posicion['precio_in']
                    mfe_p = posicion['precio_in'] - posicion['min_low']
                    pnl_p = posicion['precio_in'] - ask_
                trades.append({
                    'fecha_entrada':  posicion['fecha'],
                    'fecha_salida':   t,
                    'precio_entrada': posicion['precio_in'],
                    'precio_salida':  bid_ if es_long else ask_,
                    'pnl_price':      pnl_p,
                    'tipo':           'Cierre_Viernes',
                    'mae_price':      max(0.0, mae_p),
                    'mfe_price':      max(0.0, mfe_p),
                })
            break

        s              = evaluar_señal_mr(arr, metodo, lb)
        target_entrada = 1  if es_long else -1
        target_salida  = -1 if es_long else  1

        if s == target_salida and posicionado:
            if es_long:
                mae_p = posicion['precio_in'] - posicion['min_low']
                mfe_p = posicion['max_high']  - posicion['precio_in']
                pnl_p = bid_ - posicion['precio_in']
            else:
                mae_p = posicion['max_high']  - posicion['precio_in']
                mfe_p = posicion['precio_in'] - posicion['min_low']
                pnl_p = posicion['precio_in'] - ask_
            trades.append({
                'fecha_entrada':  posicion['fecha'],
                'fecha_salida':   t,
                'precio_entrada': posicion['precio_in'],
                'precio_salida':  bid_ if es_long else ask_,
                'pnl_price':      pnl_p,
                'tipo':           'Cierre_Señal',
                'mae_price':      max(0.0, mae_p),
                'mfe_price':      max(0.0, mfe_p),
            })
            posicionado = False
            posicion    = None

        if s == target_entrada and not posicionado:
            precio_in = ask_ if es_long else bid_
            posicion  = {
                'precio_in': precio_in,
                'fecha':     t,
                'max_high':  float(high[-1]),
                'min_low':   float(low[-1]),
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


def _metricas(df_res, capital_inicial, balance_final, reserva_final,
              comision_rt, swap_rate, year, lado):
    total = len(df_res)
    gan   = df_res[df_res['pnl_neto_usd'] > 0]  if total else pd.DataFrame()
    per   = df_res[df_res['pnl_neto_usd'] <= 0] if total else pd.DataFrame()

    gp  = gan['pnl_neto_usd'].sum() if not gan.empty else 0.0
    gl  = abs(per['pnl_neto_usd'].sum()) if not per.empty else 1e-9
    np_ = gp - gl
    hr  = len(gan) / total if total else 0
    pf  = gp / gl
    ret = np_ / capital_inicial * 100
    aw  = gan['pnl_neto_usd'].mean() if not gan.empty else 0.0
    al  = abs(per['pnl_neto_usd'].mean()) if not per.empty else 1e-9
    rr  = aw / al
    mda = df_res['drawdown_abs'].max() if total else 0.0
    mdp = df_res['drawdown_pct'].max() if total else 0.0
    sh  = calcular_sharpe(df_res['retorno_trade'].values, RISK_FREE_RATE) if total else 0.0
    cal = ret / mdp if mdp > 0 else float('inf')
    ts  = df_res['swap_usd'].sum()      if total else 0.0
    tc  = df_res['comision_usd'].sum()  if total else 0.0
    pat = df_res['patrimonio_total'].iloc[-1] if total else capital_inicial
    mg, mp = calcular_rachas(df_res['pnl_neto_usd'].values) if total else (0, 0)

    expectancy = (hr * aw) - ((1.0 - hr) * al)
    avg_mae    = df_res['mae_usd'].mean() if total else 0.0
    avg_mfe    = df_res['mfe_usd'].mean() if total else 0.0

    r = df_res['retorno_trade'].values if total else np.array([])
    if total:
        ex       = r - RISK_FREE_RATE
        downside = ex[ex < 0]
        so = (float('inf') if len(downside) < 2
              else (0.0 if downside.std(ddof=1) <= 1e-14
                    else np.sqrt(len(ex)) * ex.mean() / downside.std(ddof=1)))
        var_95  = np.percentile(r, 5)
        cvar_95 = r[r <= var_95].mean() if len(r[r <= var_95]) > 0 else var_95
    else:
        so = var_95 = cvar_95 = 0.0

    durations = []
    peak_val  = capital_inicial
    peak_time = df_res['fecha_entrada'].iloc[0] if total else None
    for _, row in df_res.iterrows():
        if row['patrimonio_total'] >= peak_val:
            peak_val  = row['patrimonio_total']
            peak_time = row['fecha_salida']
        elif peak_time is not None:
            durations.append(row['fecha_salida'] - peak_time)
    max_dd_dur_days = (max(d.total_seconds() / 86400.0 for d in durations)
                       if durations else 0.0)

    if total > 0:
        total_dur           = (df_res['fecha_salida'] - df_res['fecha_entrada']).sum()
        total_duration_days = total_dur.total_seconds() / 86400.0
        avg_duration_hours  = total_dur.total_seconds() / 3600.0 / total
    else:
        total_duration_days = avg_duration_hours = 0.0

    if total >= 3:
        mean_r, std_r = np.mean(r), np.std(r, ddof=1)
        if std_r > 1e-14:
            skew_val = np.sum(((r - mean_r) / std_r) ** 3) / total
            kurt_val = np.sum(((r - mean_r) / std_r) ** 4) / total - 3.0
        else:
            skew_val = kurt_val = 0.0
    else:
        skew_val = kurt_val = 0.0

    w = 60
    print(f"\n{'═'*w}")
    print(f"   BACKTEST MR DIRECTO {lado.upper()} {year}  —  {SYMBOL}".center(w))
    print(f"{'═'*w}")
    print(f"  Capital Inicial               : ${capital_inicial:>12,.2f}")
    print(f"  Comisión RT                    : ${comision_rt:>12,.4f} / lote")
    print(f"  Swap {lado[:5]:<5}                   : ${swap_rate:>12,.4f} / lote-noche")
    print(f"{'─'*w}")
    print(f"  Ganancias Apartadas           : ${reserva_final:>12,.2f}")
    print(f"  Capital Operativo Final       : ${balance_final:>12,.2f}")
    print(f"  Patrimonio Total Final        : ${pat:>12,.2f}")
    print(f"{'─'*w}")
    print(f"  Beneficio Neto                : ${np_:>+12,.2f}")
    print(f"  Retorno sobre Capital         : {ret:>+11.2f}%")
    print(f"  Esperanza Matemática (Trade)  : ${expectancy:>12,.2f}")
    print(f"  Profit Factor                 : {pf:>12.3f}")
    print(f"  Risk / Reward                 : {rr:>12.3f}")
    print(f"{'─'*w}")
    print(f"  Sharpe Ratio                  : {sh:>12.4f}")
    print(f"  Sortino Ratio                 : {so:>12.4f}")
    print(f"  Calmar Ratio                  : {cal:>12.4f}")
    print(f"  Value at Risk (95%)           : {var_95*100:>11.2f}%")
    print(f"  Conditional VaR (95%)         : {cvar_95*100:>11.2f}%")
    print(f"{'─'*w}")
    print(f"  Avg. MAE (Riesgo Latente)     : ${avg_mae:>12,.2f}")
    print(f"  Avg. MFE (Beneficio Latente)  : ${avg_mfe:>12,.2f}")
    print(f"{'─'*w}")
    print(f"  Max Drawdown ($)              : -${mda:>11,.2f}")
    print(f"  Max Drawdown (%)              : -{mdp:>10.2f}%")
    print(f"  Max DD Duration (Recuperación): {max_dd_dur_days:>11.2f} días")
    print(f"  Tiempo Total Expuesto         : {total_duration_days:>11.2f} días")
    print(f"  Duración Promedio Trade       : {avg_duration_hours:>11.2f} horas")
    print(f"{'─'*w}")
    print(f"  Asimetría (Skewness)          : {skew_val:>12.2f}")
    print(f"  Curtosis de Exceso (Kurtosis) : {kurt_val:>12.2f}")
    print(f"{'─'*w}")
    print(f"  Total Operaciones             : {total:>12}")
    print(f"  Ganadoras                     : {len(gan):>12}")
    print(f"  Perdedoras                    : {len(per):>12}")
    print(f"  Hit Ratio                     : {hr:>12.4f} ({hr*100:.2f}%)")
    print(f"{'═'*w}")

    return {
        'year': year, 'lado': lado,
        'total_trades': total, 'ganadoras': len(gan), 'perdedoras': len(per),
        'hit_ratio_pct': round(hr*100, 2),
        'gross_profit': round(gp, 2), 'gross_loss': round(gl, 2),
        'net_profit': round(np_, 2), 'retorno_pct': round(ret, 2),
        'profit_factor': round(pf, 3), 'risk_reward': round(rr, 3),
        'avg_win': round(aw, 2), 'avg_loss': round(al, 2),
        'sharpe_ratio': round(sh, 4), 'calmar_ratio': round(cal, 4),
        'max_dd_abs': round(mda, 2), 'max_dd_pct': round(mdp, 2),
        'total_swap': round(ts, 2), 'total_comisiones': round(tc, 2),
        'patrimonio_final': round(pat, 2),
        'racha_max_ganadora': mg, 'racha_max_perdedora': mp,
        'expectancy': round(expectancy, 2),
        'avg_mae': round(avg_mae, 2), 'avg_mfe': round(avg_mfe, 2),
        'sortino_ratio': round(so, 4) if so != float('inf') else float('inf'),
        'var_95_pct': round(var_95 * 100, 2),
        'cvar_95_pct': round(cvar_95 * 100, 2),
        'max_dd_duration': round(max_dd_dur_days, 2),
        'tiempo_expuesto': round(total_duration_days, 2),
        'avg_duracion_hrs': round(avg_duration_hours, 2),
        'skewness': round(skew_val, 2),
        'kurtosis': round(kurt_val, 2),
    }


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
        datetime(year-1, 12, 1), datetime(year, 12, 28)
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"[{year}] Sin datos"); return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
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

        buf_l = lunes - timedelta(minutes=(lb_l + 25) * vela_l)
        buf_s = lunes - timedelta(minutes=(lb_s + 25) * vela_s)

        sem_l = df.loc[buf_l:viernes]
        sem_s = df.loc[buf_s:viernes]

        if not sem_l.empty:
            trades_long  += _semana_lado(
                sem_l, lunes, viernes, met_l, lb_l, vela_l, es_long=True)
        if not sem_s.empty:
            trades_short += _semana_lado(
                sem_s, lunes, viernes, met_s, lb_s, vela_s, es_long=False)

    res_long = res_short = None
    df_l_out = df_s_out  = None

    if trades_long:
        df_l, bal_l, rev_l = _contabilizar(
            trades_long, CAPITAL_LONG, swap_long_r, swap_mode,
            contract_size, tick_value, tick_size, rollover3days, comision_rt)
        csv_l = os.path.expanduser(f"~/{SYMBOL}_{year}_mr_directo_long.csv")
#        df_l.to_csv(csv_l)
        res_long = _metricas(df_l, CAPITAL_LONG, bal_l, rev_l,
                             comision_rt, swap_long_r, year, "LONG")
        df_l_out = df_l
        print(f"  CSV guardado: {csv_l}")
    else:
        print(f"[{year}] Sin operaciones LONG.")

    if trades_short:
        df_s, bal_s, rev_s = _contabilizar(
            trades_short, CAPITAL_SHORT, swap_short_r, swap_mode,
            contract_size, tick_value, tick_size, rollover3days, comision_rt)
        csv_s = os.path.expanduser(f"~/{SYMBOL}_{year}_mr_directo_short.csv")
        #df_s.to_csv(csv_s)
        res_short = _metricas(df_s, CAPITAL_SHORT, bal_s, rev_s,
                              comision_rt, swap_short_r, year, "SHORT")
        df_s_out = df_s
        print(f"  CSV guardado: {csv_s}")
    else:
        print(f"[{year}] Sin operaciones SHORT.")

    if df_l_out is not None or df_s_out is not None:
        _generar_equity_chart(df_l_out, df_s_out, year, SYMBOL)

    return res_long, res_short

def imprimir_tabla(resultados: list, lado: str):
    rs = [r for r in resultados if r and r.get('lado') == lado]
    if not rs:
        return

    years = [str(r['year']) for r in rs]
    filas = [
        ("Patrimonio Final ($)",     "patrimonio_final",    "$ {:>14,.2f}"),
        ("Beneficio Neto ($)",       "net_profit",          "${:>+14,.2f}"),
        ("Retorno (%)",              "retorno_pct",          "{:>+13.2f} %"),
        ("Gross Profit ($)",         "gross_profit",        "$ {:>14,.2f}"),
        ("Gross Loss ($)",           "gross_loss",          "$ {:>14,.2f}"),
        ("Profit Factor",            "profit_factor",       "{:>15.3f}"),
        ("Risk / Reward",            "risk_reward",         "{:>15.3f}"),
        ("Esperanza Matemática ($)", "expectancy",          "$ {:>14,.2f}"),
        ("Sharpe Ratio",             "sharpe_ratio",        "{:>15.4f}"),
        ("Sortino Ratio",            "sortino_ratio",       "{:>15.4f}"),
        ("Calmar Ratio",             "calmar_ratio",        "{:>15.4f}"),
        ("VaR 95% (%)",              "var_95_pct",          "{:>13.2f} %"),
        ("cVaR 95% (%)",             "cvar_95_pct",         "{:>13.2f} %"),
        ("Max Drawdown ($)",         "max_dd_abs",          "-${:>13,.2f}"),
        ("Max Drawdown (%)",         "max_dd_pct",          "-{:>13.2f} %"),
        ("Max DD Duration (Días)",   "max_dd_duration",     "{:>15.2f}"),
        ("Avg MAE (Riesgo Lat. $)",  "avg_mae",             "$ {:>14,.2f}"),
        ("Avg MFE (Benef. Lat. $)",  "avg_mfe",             "$ {:>14,.2f}"),
        ("Total Operaciones",        "total_trades",        "{:>15}"),
        ("Ganadoras",                "ganadoras",           "{:>15}"),
        ("Perdedoras",               "perdedoras",          "{:>15}"),
        ("Hit Ratio (%)",            "hit_ratio_pct",       "{:>14.2f} %"),
        ("Avg Ganancia / op ($)",    "avg_win",             "$ {:>14,.2f}"),
        ("Avg Pérdida / op ($)",     "avg_loss",            "$ {:>14,.2f}"),
        ("Racha Máx. Ganadoras",     "racha_max_ganadora",  "{:>15}"),
        ("Racha Máx. Perdedoras",    "racha_max_perdedora", "{:>15}"),
        ("Tiempo Expuesto (Días)",   "tiempo_expuesto",     "{:>15.2f}"),
        ("Avg Duración Trade (Hrs)", "avg_duracion_hrs",    "{:>15.2f}"),
        ("Asimetría (Skewness)",     "skewness",            "{:>15.2f}"),
        ("Curtosis",                 "kurtosis",            "{:>15.2f}"),
        ("Swap Total ($)",           "total_swap",          "$ {:>14,.2f}"),
        ("Comisiones Total ($)",     "total_comisiones",    "$ {:>14,.2f}"),
    ]
    secciones = {
        "Patrimonio Final ($)":    "── Rendimiento ──",
        "Sharpe Ratio":            "── Ratios de Eficiencia Financiera ──",
        "Max Drawdown ($)":        "── Riesgo de Caída (Drawdown) ──",
        "Avg MAE (Riesgo Lat. $)": "── Métricas de Ejecución (MAE/MFE) ──",
        "Total Operaciones":       "── Operaciones y Rachas ──",
        "Tiempo Expuesto (Días)":  "── Distribución y Tiempos ──",
        "Swap Total ($)":          "── Costos Operativos ──",
    }

    CL, CY = 28, 18
    W  = CL + CY * len(years) + 2
    hd = "".join(f"{y:>{CY}}" for y in years)
    cap = CAPITAL_LONG if lado == "LONG" else CAPITAL_SHORT

    print(f"\n\n{'═'*W}")
    print(f"  COMPARATIVO AÑO A AÑO  —  {lado}".center(W))
    print(f"  {SYMBOL}  |  Capital: ${cap:,.0f}  |  Mean Reversion Directo".center(W))
    print(f"{'═'*W}")
    print(f"  {'MÉTRICA':<{CL-2}}{hd}")
    print(f"{'─'*W}")

    for etiq, clave, fmt in filas:
        if etiq in secciones:
            print(f"{'─'*W}")
            print(f"  {secciones[etiq]}")
            print(f"{'─'*W}")
        vals = ""
        for r in rs:
            v = r.get(clave, 0)
            try:
                c = fmt.format(v)
            except (TypeError, ValueError):
                c = f"{str(v):>{CY}}"
            vals += f"{c:>{CY}}"
        print(f"  {etiq:<{CL-2}}{vals}")

    print(f"{'═'*W}")


if __name__ == "__main__":
    todos_long  = []
    todos_short = []

    for year in YEARS:
        print(f"\n{'#'*65}")
        print(f"  Backtest MR Directo {year}  —  {SYMBOL}")
        print(f"{'#'*65}")
        r_long, r_short = backtest_año(year)
        if r_long:  todos_long.append(r_long)
        if r_short: todos_short.append(r_short)

    imprimir_tabla(todos_long,  "LONG")
    imprimir_tabla(todos_short, "SHORT")
