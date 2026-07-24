$MON      = 100
$LEVERAGE = 8
$ST_MON   = "true"


$instancias = @(
    @{ symbol = "AUDCAD_"; short = "false" }
)


$rutaActual = Get-Location

$cmdTerminal  = ""

for ($i = 0; $i -lt $instancias.Count; $i++) {
    $instancia = $instancias[$i]

    $direction = if ($instancia.short -eq "true") { "SHORT" } else { "LONG" }
    $tituloPestana = "$($instancia.symbol)_$direction"

    $argsPython = "cd .. ; python bot_normal.py --symbol=$($instancia.symbol) --short=$($instancia.short) --mon=$MON --leverage=$LEVERAGE --st_mon=$ST_MON"

    $comandoPestana = "-p `"Windows PowerShell`" -d `"$rutaActual`" --title `"$tituloPestana`" powershell -NoExit -Command `"$argsPython`""

    if ($i -eq 0) {
        $cmdTerminal = $comandoPestana
    } else {
        $cmdTerminal += " ; new-tab $comandoPestana"
    }
}

Start-Process wt -ArgumentList $cmdTerminal
