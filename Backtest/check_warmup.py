"""Sanidad de ta_warmup.candles_requeridas.

La sonda mide empiricamente el periodo inestable de TA-Lib. Si se rompe en
silencio devolveria un numero enorme y _semana_lado saltaria todas las semanas,
que es exactamente el fallo que veniamos a corregir. Se contrasta contra las
formulas cerradas conocidas.

Requiere TA-Lib.

    python Backtest/check_warmup.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ta_warmup import velas_minimas, candles_requeridas

# Primer indice no-NaN + 1 = velas necesarias
FORMULAS = {
    "SMA":      lambda n: n,
    "EMA":      lambda n: n,
    "WMA":      lambda n: n,
    "TRIMA":    lambda n: n,
    "MIDPOINT": lambda n: n,
    "KAMA":     lambda n: n + 1,
    "DEMA":     lambda n: 2 * (n - 1) + 1,
    "TEMA":     lambda n: 3 * (n - 1) + 1,
    "T3":       lambda n: 6 * (n - 1) + 1,
}

LOOKBACKS = (5, 30, 110)


def main() -> int:
    fallos = 0
    print(f"  {'metodo':<10}{'lb':>5}{'sonda':>8}{'formula':>9}{'+margen':>9}")
    for metodo, formula in FORMULAS.items():
        for lb in LOOKBACKS:
            medido   = velas_minimas(metodo, lb)
            esperado = formula(lb)
            con_marg = candles_requeridas(metodo, lb)
            marca = "" if medido == esperado else "   <-- FALLO"
            if medido != esperado:
                fallos += 1
            print(f"  {metodo:<10}{lb:>5}{medido:>8}{esperado:>9}{con_marg:>9}{marca}")

    peor = max(candles_requeridas(m, 110) for m in FORMULAS)
    print(f"\n  Peor caso del espacio de busqueda (lb=110): {peor} velas"
          f" -> {peor * 100:,} filas M1 con vela=100")

    if fallos:
        print(f"FALLO: {fallos} discrepancias con las formulas cerradas.")
        return 1
    print("OK: la sonda coincide con las formulas de TA-Lib.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
