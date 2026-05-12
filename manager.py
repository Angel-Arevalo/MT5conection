import json
import os
import time
import MetaTrader5 as mt5
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

JSON_FILE: str = "magic_numbers.json"
CAPITAL_INICIAL: float = 1000.0
APALANCAMIENTO: float = 40.0

def get_or_create_magic(symbol: str, is_short: bool) -> int:
    direccion: str = "SHORT" if is_short else "LONG"
    clave: str = f"{symbol}_{direccion}"

    _: int
    for _ in range(10):
        try:
            data: Dict[str, Any]
            if os.path.exists(JSON_FILE):
                with open(JSON_FILE, "r") as f:
                    data = json.load(f)
            else:
                data = {"last_magic": 999, "configs": {}}

            if clave in data["configs"]:
                return int(data["configs"][clave])

            nuevo_magic: int = int(data["last_magic"]) + 1
            data["last_magic"] = nuevo_magic
            data["configs"][clave] = nuevo_magic

            with open(JSON_FILE, "w") as f:
                json.dump(data, f, indent=4)

            print(f"[MANAGER] Nuevo Magic Number asignado -> {clave}: {nuevo_magic}")
            return nuevo_magic

        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.2)

    raise Exception(f"No se pudo acceder a {JSON_FILE} tras varios intentos.")

def calcular_volumen_estricto(symbol: str, magic_number: int, is_short: bool) -> float:
    info: Any = mt5.symbol_info(symbol)
    if info is None:
        return 0.0

    from_date: datetime = datetime(2000, 1, 1, tzinfo=timezone.utc)
    to_date: datetime = datetime.now(timezone.utc)

    deals: Optional[Tuple[Any, ...]] = mt5.history_deals_get(from_date, to_date)
    total_pnl: float = 0.0

    if deals:
        deals_bot: List[Any] = [d for d in deals if d.magic == magic_number]
        deal: Any
        for deal in deals_bot:
            total_pnl += float(deal.profit + deal.commission + deal.fee + deal.swap)

    balance_virtual: float = CAPITAL_INICIAL + total_pnl
    balance_operativo: float = min(CAPITAL_INICIAL, balance_virtual)

    if balance_operativo <= 0.0:
        print(f"[MANAGER] Bot {magic_number} quebrado. Balance: {balance_operativo:.2f}")
        return 0.0

    tipo_orden: int = mt5.ORDER_TYPE_SELL if is_short else mt5.ORDER_TYPE_BUY
    precio_ref: float = float(info.bid if is_short else info.ask)

    margin_1lot: Optional[float] = mt5.order_calc_margin(tipo_orden, symbol, 1.0, precio_ref)
    if not margin_1lot or margin_1lot == 0.0:
        return 0.0

    nocional_disponible: float = balance_operativo * APALANCAMIENTO
    lote_exacto: float = nocional_disponible / margin_1lot
    lote_redondeado: float = float((lote_exacto // info.volume_step) * info.volume_step)

    return float(max(info.volume_min, min(lote_redondeado, info.volume_max)))
