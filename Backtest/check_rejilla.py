"""Regresion critica: la rejilla muestreada no se movio.

_semana_lado pasó de reconstruir un DataFrame por iteracion

    w = df_semana.iloc[:i+1].iloc[::-1][::vela_min].iloc[::-1]

a un unico precalculo posicional

    sel = np.arange(start % vela_min, len(df_semana), vela_min)

Si ambos no coinciden exactamente, la ventana por filas movio la rejilla y todos
los numeros aguas abajo son invalidos. No requiere MT5 ni TA-Lib.

    python Backtest/check_rejilla.py
"""

import numpy as np
import pandas as pd

CASOS = [
    # (filas M1 en la ventana, vela_min, offsets de lunes a probar)
    (5_000,  100, (0, 1, 99, 100, 302, 1_666)),
    (3_000,   85, (0, 7, 84, 85, 257,   999)),
    (12_000,  54, (0, 1, 53, 54, 164, 4_000)),
    (999,      7, (0, 3,  6,  7,  23,   333)),
    (1_234,    1, (0, 1,  5, 17, 100,   411)),
    (2_000,  110, (0, 9, 109, 110, 331,  666)),
]


def main() -> int:
    fallos = 0
    comparaciones = 0

    for n, vela, starts in CASOS:
        df = pd.DataFrame({'bid': np.arange(n, dtype=float)})
        for start in starts:
            if start >= n:
                continue
            sel = np.arange(start % vela, n, vela)

            for i in range(start, n, vela):
                viejo = df.iloc[:i + 1].iloc[::-1][::vela].iloc[::-1].index.values
                k = (i - start % vela) // vela
                nuevo = sel[:k + 1]
                comparaciones += 1
                if not np.array_equal(viejo, nuevo):
                    fallos += 1
                    print(f"  MISMATCH n={n} vela={vela} start={start} i={i}")
                    break

            # el lunes debe caer exactamente en la rejilla
            if start not in sel:
                fallos += 1
                print(f"  LUNES FUERA DE REJILLA n={n} vela={vela} start={start}")
            elif int(np.searchsorted(sel, start)) != start // vela:
                fallos += 1
                print(f"  K_LUNES INCORRECTO n={n} vela={vela} start={start}")

        print(f"  n={n:>6} vela={vela:>4}  ok")

    print(f"\n{comparaciones} comparaciones, {fallos} fallos")
    if fallos:
        print("FALLO: la rejilla se movio. No confiar en el rerun.")
        return 1
    print("OK: rejilla invariante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
