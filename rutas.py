"""Localiza el submodulo optimal-moving-average.

MT5connection/.gitmodules y el .gitmodules del repo padre declaran el mismo
submodulo contra el mismo remoto, asi que puede estar clonado en cualquiera de
los dos sitios. Inicializar los dos crearia dos copias que se separan con el
tiempo (un arreglo aplicado en una no llega a la otra), asi que se usa la que
este disponible y se prefiere la local.
"""

import os

_AQUI = os.path.dirname(os.path.abspath(__file__))

_CANDIDATAS = (
    os.path.join(_AQUI, 'optimal-moving-average'),                        # submodulo propio
    os.path.join(os.path.dirname(_AQUI), 'optimal-moving-average'),       # copia del repo padre
)


def ruta_oma() -> str:
    """Ruta al checkout de optimal-moving-average que si tiene contenido."""
    for candidata in _CANDIDATAS:
        if os.path.isfile(os.path.join(candidata, 'find_best.py')):
            return candidata

    raise ImportError(
        "No se encontro find_best.py en:\n  " + "\n  ".join(_CANDIDATAS) +
        "\nEjecuta 'git submodule update --init' en MT5connection, o clona "
        "optimal-moving-average junto al repo."
    )
