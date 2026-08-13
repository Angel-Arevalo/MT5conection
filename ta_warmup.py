"""Periodo inestable real de cada media movil de TA-Lib.

Cada metodo de TA-Lib necesita una cantidad distinta de velas antes de emitir un
valor no-NaN: SMA/EMA/WMA/TRIMA/MIDPOINT necesitan lb, KAMA lb+1, DEMA 2(lb-1)+1,
TEMA 3(lb-1)+1 y T3 6(lb-1)+1. El optimizador evalua sobre 4 semanas remuestreadas
(~380-1150 velas) asi que casi nunca choca con ese limite, pero los consumidores
(el backtest y el bot en vivo) trabajan con ~200 velas y se quedan en NaN toda la
semana sin avisar.

Se mide empiricamente en vez de codificar las formulas: es auto-correctivo si
cambia la version de TA-Lib.
"""

import numpy as np
import talib
from typing import Callable, Dict

FAST_METHODS: Dict[str, Callable] = {
    "SMA": talib.SMA, "EMA": talib.EMA, "WMA": talib.WMA,
    "DEMA": talib.DEMA, "TEMA": talib.TEMA, "TRIMA": talib.TRIMA,
    "KAMA": talib.KAMA, "T3": talib.T3, "MIDPOINT": talib.MIDPOINT,
}

_LB_CACHE: Dict[tuple, int] = {}


def velas_minimas(metodo: str, lb: int) -> int:
    """Primer largo de entrada con el que el metodo devuelve un ultimo valor no-NaN."""
    key = (metodo, lb)
    if key in _LB_CACHE:
        return _LB_CACHE[key]

    f = FAST_METHODS[metodo]
    n_max = 7 * lb + 64                                   # supera el 6*(lb-1)+1 de T3

    # Serie no constante y deterministica: KAMA degenera con entrada plana.
    rng = np.random.default_rng(0)
    x = 1.0 + np.cumsum(rng.normal(0, 1e-3, n_max)) + np.linspace(0, 0.5, n_max)

    need = n_max
    for n in range(lb, n_max + 1):
        if not np.isnan(f(x[:n], timeperiod=lb)[-1]):
            need = n
            break

    _LB_CACHE[key] = need
    return need


def candles_requeridas(metodo: str, lb: int, margen: int = 5) -> int:
    """Velas necesarias para una MA valida, con margen de seguridad."""
    return velas_minimas(metodo, lb) + margen
