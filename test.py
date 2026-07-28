import time
from datetime import datetime
from DataIterator import DataIterator

fecha_inicio = datetime(2026, 6, 15)
fecha_fin = datetime(2026, 6, 16)

iterador = DataIterator("EURUSD_")

print(iterador.market)

iterador.backtest(fecha_inicio, fecha_fin)

while True:
    print(iterador.next_candle())
