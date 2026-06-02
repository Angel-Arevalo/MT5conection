MON=200
LEVERAGE=200
ST_MON="true"
DISPLAY_NUM=":100"
SESSION_NAME="trading_bots"

RUTA_MT5="$HOME/.wine/drive_c/Program Files/CFI2 MetaTrader 5 Terminal/terminal64.exe"

instancias=(
  "US500_SPOT:false"
  "US500_SPOT:true"
  "GER30_SPOT:false"
  "GER30_SPOT:true"
  "EUR50_SPOT:false"
  "EUR50_SPOT:true"
)

if ! pgrep -x "Xvfb" >/dev/null; then
  echo "[+] Iniciando Xvfb en el display $DISPLAY_NUM (Pantalla virtual)..."
  Xvfb $DISPLAY_NUM -screen 0 1024x768x16 &
  sleep 2
else
  echo "[~] Xvfb ya está corriendo."
fi

export DISPLAY=$DISPLAY_NUM
export WINEDEBUG=-all

if ! pgrep -f "terminal64.exe" >/dev/null; then
  echo "[+] Iniciando MetaTrader 5 bajo Wine en segundo plano..."
  wine "$RUTA_MT5" &
  sleep 5
else
  echo "[~] MetaTrader 5 ya está abierto en Wine."
fi

echo "[+] Limpiando sesiones previas de tmux..."
tmux kill-session -t $SESSION_NAME 2>/dev/null

echo "[+] Creando nueva sesión de tmux: $SESSION_NAME"

tmux new-session -d -s $SESSION_NAME -n "principal"

for instancia in "${instancias[@]}"; do

  IFS=":" read -r symbol short <<<"$instancia"

  if [ "$short" == "true" ]; then
    direction="SHORT"
  else
    direction="LONG"
  fi

  WINDOW_NAME="${symbol}_${direction}"
  echo "  -> Creando pestaña para $WINDOW_NAME"

  tmux new-window -t $SESSION_NAME -n "$WINDOW_NAME"

  CMD="export DISPLAY=$DISPLAY_NUM; export WINEDEBUG=-all; cd .. && wine python bot_ADX.py --symbol=$symbol --short=$short --mon=$MON --leverage=$LEVERAGE --st_mon=$ST_MON"

  tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "$CMD" Enter
done

tmux select-window -t "$SESSION_NAME:1"

echo "[V] ¡Proceso completado con éxito!"
