$MON      = 200
$LEVERAGE = 200
$ST_MON   = "true"


$instancias = @(
    @{ symbol = "US500_SPOT"; short = "false" },
    @{ symbol = "US500_SPOT"; short = "true"  },
    @{ symbol = "CAC40_SPOT"; short = "false" },
    @{ symbol = "CAC40_SPOT"; short = "true"  },
    @{ symbol = "GBPUSD_";    short = "false" },
    @{ symbol = "GBPUSD_";    short = "true"  },
    @{ symbol = "USDJPY_";    short = "false" },
    @{ symbol = "USDJPY_";    short = "true"  },
    @{ symbol = "USDCAD_";    short = "false" },
    @{ symbol = "USDCAD_";    short = "true"  },
    @{ symbol = "GER30_SPOT"; short = "false" },
    @{ symbol = "GER30_SPOT"; short = "true"  },
    @{ symbol = "EUR50_SPOT"; short = "false" },
    @{ symbol = "EUR50_SPOT"; short = "true"  },
    @{ symbol = "AUDCAD_";    short = "false" },
    @{ symbol = "AUDCAD_";    short = "true"  },
    @{ symbol = "EURGBP_";    short = "false" },
    @{ symbol = "EURGBP_";    short = "true"  },
    @{ symbol = "AUDNZD_";    short = "false" },
    @{ symbol = "AUDNZD_";    short = "true"  }
)


$rutaActual = Get-Location

$cmdTerminal  = ""

for ($i = 0; $i -lt $instancias.Count; $i++) {
    $instancia = $instancias[$i]

    $direction = if ($instancia.short -eq "true") { "SHORT" } else { "LONG" }
    $tituloPestana = "$($instancia.symbol)_$direction"

    $argsPython = "cd .. ; python bot_ADX.py --symbol=$($instancia.symbol) --short=$($instancia.short) --mon=$MON --leverage=$LEVERAGE --st_mon=$ST_MON"

    $comandoPestana = "-p `"Windows PowerShell`" -d `"$rutaActual`" --title `"$tituloPestana`" powershell -NoExit -Command `"$argsPython`""

    if ($i -eq 0) {
        $cmdTerminal = $comandoPestana
    } else {
        $cmdTerminal += " ; new-tab $comandoPestana"
    }
}

Start-Process wt -ArgumentList $cmdTerminal
