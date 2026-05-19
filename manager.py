import json
import os
import requests
import time
import MetaTrader5 as mt5
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List, Literal

JSON_FILE: str = "magic_numbers.json"
CAPITAL_INICIAL: float = 1000.0
APALANCAMIENTO: float = 40.0

TELEGRAM_BOT_TOKEN: str = "8890946032:AAF4hDUq08qxy1p6M1v878zFmTwSB1WMLo8"
TELEGRAM_CHAT_ID: str = "6045302342"

Direction = Literal["LONG", "SHORT", "BOTH"]

def get_or_create_magic(symbol: str, direction: Direction) -> int:
    clave: str = f"{symbol}_{direction}"

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


def _margen_por_lote(symbol: str, direction: Direction, precio_ask: float, precio_bid: float) -> float:
    def _calc(tipo: int, precio: float) -> float:
        m = mt5.order_calc_margin(tipo, symbol, 1.0, precio)
        return float(m) if m and m > 0.0 else 0.0

    if direction == "LONG":
        return _calc(mt5.ORDER_TYPE_BUY, precio_ask)

    if direction == "SHORT":
        return _calc(mt5.ORDER_TYPE_SELL, precio_bid)

    return max(
        _calc(mt5.ORDER_TYPE_BUY,  precio_ask),
        _calc(mt5.ORDER_TYPE_SELL, precio_bid),
    )


def calcular_volumen_estricto(
    symbol: str,
    magic_number: int,
    direction: Direction = "LONG",
    *,
    capital: float = CAPITAL_INICIAL, 
    apalancamiento: float = APALANCAMIENTO, 
    ignorar_historial: bool = False) -> float:
    info: Any = mt5.symbol_info(symbol)
    if info is None:
        return 0.0

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0

    precio_ask: float = float(tick.ask)
    precio_bid: float = float(tick.bid)

    if ignorar_historial:
        balance_operativo: float = capital
    else:
        from_date: datetime = datetime(2000, 1, 1, tzinfo=timezone.utc)
        to_date:   datetime = datetime.now(timezone.utc)

        deals: Optional[Tuple[Any, ...]] = mt5.history_deals_get(from_date, to_date)
        total_pnl: float = 0.0

        if deals:
            deals_bot: List[Any] = [d for d in deals if d.magic == magic_number]
            for deal in deals_bot:
                total_pnl += float(
                    deal.profit + deal.commission + deal.fee + deal.swap
                )

        balance_virtual:  float = capital + total_pnl
        balance_operativo = min(capital, balance_virtual)

    if balance_operativo <= 0.0:
        print(
            f"[MANAGER] Bot {magic_number} quebrado. "
            f"Balance operativo: {balance_operativo:.2f}"
        )
        return 0.0

    notional_por_lote: float = float(info.trade_contract_size) * precio_ask
    if notional_por_lote <= 0.0:
        return 0.0

    lote_exacto: float = (balance_operativo * apalancamiento) / notional_por_lote

    margin_1lot: float = _margen_por_lote(symbol, direction, precio_ask, precio_bid)
    if margin_1lot > 0.0:
        margen_requerido: float = lote_exacto * margin_1lot
        if margen_requerido > balance_operativo:
            lote_exacto = balance_operativo / margin_1lot
            print(
                f"[MANAGER] Volumen reducido por margen insuficiente: "
                f"{lote_exacto:.4f} lotes (margen máx: {balance_operativo:.2f})"
            )

    lote_redondeado: float = float(
        (lote_exacto // info.volume_step) * info.volume_step
    )
    return float(max(info.volume_min, min(lote_redondeado, info.volume_max)))

def enviar_mensaje_telegram(symbol: str, direction: Direction, is_close: bool, magic_number: int) -> None:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "TU_TOKEN_DE_BOTFATHER":
        print("[TELEGRAM] Credenciales no configuradas. Mensaje omitido.")
        return


    accion_str = "🔴 *CIERRE* de" if is_close else "🟢 *APERTURA* de"
    direccion_str = "📈 LONG" if direction == "LONG" else ("📉 SHORT" if direction == "SHORT" else "🔄 BOTH")

    mensaje = (
        f"🤖 *Notificación de Trading Bot*\n\n"
        f"Acción: {accion_str} {direccion_str}\n"
        f"Activo: *{symbol}*\n"
        f"Magic Number: `{magic_number}`\n"
        f"Hora (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown"
    }

    try:

        response = requests.post(url, json=payload, timeout=5.0)

        if response.status_code == 200:
            print(f"[TELEGRAM] Mensaje enviado correctamente para {symbol} (Magic: {magic_number})")
        else:
            print(f"[TELEGRAM] Error de la API de Telegram ({response.status_code}): {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"[TELEGRAM] Excepción de red al intentar enviar mensaje: {e}")
