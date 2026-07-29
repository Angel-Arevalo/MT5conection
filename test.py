import time
from datetime import datetime
from DataIterator import DataIterator

fecha_inicio = datetime(2026, 6, 15)
fecha_fin = datetime(2026, 6, 18)

iterador = DataIterator("BTCUSD")

print(iterador.market)

iterador.backtest(fecha_fin, fecha_fin)

while True:
    print(iterador.next_candle())
