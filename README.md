# manager.py

Gestión centralizada de **magic numbers** y **cálculo de volumen** para bots de trading en MetaTrader 5.

## Funciones principales

### `get_or_create_magic(symbol, direction)`

Asigna un **magic number único** a cada combinación de símbolo + dirección. Persiste en `magic_numbers.json`.

**Parámetros:**
- `symbol`: Instrumento (`"EURUSD"`, `"BTCUSD"`, etc.)
- `direction`: `"LONG"`, `"SHORT"` o `"BOTH"`

**Ejemplo:**
```python
magic_eurusd_long  = get_or_create_magic("EURUSD", "LONG")   # -> 1000
magic_eurusd_short = get_or_create_magic("EURUSD", "SHORT")  # -> 1001
magic_eurusd_both  = get_or_create_magic("EURUSD", "BOTH")   # -> 1002
```

Cada bot tiene su propio historial de P&L aislado del resto.

---

### `calcular_volumen_estricto(symbol, magic_number, direction, *, capital, apalancamiento, ignorar_historial)`

Calcula cuántos **lotes** operar respetando capital y apalancamiento.

**Parámetros posicionales:**
- `symbol`: Instrumento
- `magic_number`: ID del bot
- `direction`: `"LONG"`, `"SHORT"` o `"BOTH"`

**Parámetros keyword-only (opcionales):**
- `capital`: Capital base (default: `CAPITAL_INICIAL = 1000.0`)
- `apalancamiento`: Multiplicador de exposición (default: `APALANCAMIENTO = 40.0`)
- `ignorar_historial`: Si `True`, usa siempre el capital completo sin descontar pérdidas históricas (default: `False`)

**Lógica:**

1. **Balance operativo:**
   - Si `ignorar_historial=False`: `min(capital, capital + PnL_histórico)` — el bot no puede operar más capital del inicial aunque tenga ganancias
   - Si `ignorar_historial=True`: siempre opera con `capital` sin mirar el historial

2. **Cálculo de lotes:**
   ```
   nocional_objetivo = balance_operativo × apalancamiento
   lotes = nocional_objetivo / (contract_size × precio)
   ```

3. **Validación de margen:**
   - Calcula el margen real requerido con `order_calc_margin`
   - Si el margen supera el balance operativo, reduce automáticamente los lotes
   - Para `direction="BOTH"`, usa el margen más conservador (max entre BUY y SELL)

**Ejemplo básico:**
```python
volumen = calcular_volumen_estricto("EURUSD", 1000, "LONG")
# -> 0.36 lotes (con capital=$1000, apalancamiento=40x, EURUSD≈1.10)
```

**Ejemplo avanzado:**
```python
volumen = calcular_volumen_estricto(
    "BTCUSD",
    1001,
    "SHORT",
    capital=5000,              # capital personalizado
    apalancamiento=20,         # apalancamiento reducido
    ignorar_historial=True     # siempre opera con $5000
)
```

---

## Configuración global

```python
CAPITAL_INICIAL = 1000.0   # Capital base por bot
APALANCAMIENTO  = 40.0     # Exposición máxima (40x)
JSON_FILE       = "magic_numbers.json"
```

## Uso en bot.py

```python
import manager

# Inicialización
DIRECTION = "SHORT" if IS_SHORT else "LONG"
MAGIC_NUMBER = manager.get_or_create_magic(SYMBOL, DIRECTION)

# En ejecutar_orden():
volumen = manager.calcular_volumen_estricto(
    SYMBOL,
    MAGIC_NUMBER,
    DIRECTION
)
```
