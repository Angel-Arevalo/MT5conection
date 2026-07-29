class Signal:
    # Se asume que True es long y que False es short
    __long: bool
    __lot: float

    __open: bool

    def __init__(self, direction: bool, lot: float) -> None:
        self.__long = direction

        self.__lot = lot
        self.__open = True

    @property
    def long(self) -> bool:
        return self.__long

    @property
    def lot(self) -> float:
        return self.__lot

    @property
    def open(self) -> bool:
        return self.__open

    def change_type(self, id: bytes) -> None:
        self.__open = False
