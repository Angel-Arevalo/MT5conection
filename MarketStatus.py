from datetime import time, datetime, timezone
from typing import Optional

class MarketStatus:

    _in_market: bool
    _start_day: time

    _end_day: time
    _weekends: bool

    __passbyte: bytes
    __minute_to_market_close: int

    _last_updt: Optional[datetime]

    def __init__(self, start_day: time, end_day: time, weekends: bool, passwd: bytes) -> None:
        self._start_day = start_day
        self._end_day = end_day
        self._weekends = weekends
        self.__passbyte = passwd

        self._in_market = False
        self.__minute_to_market_close = 0

        self._last_updt = None

    @property
    def start_day(self) -> time:
        return self._start_day

    @property
    def end_day(self) -> time:
        return self._end_day

    @property
    def weekends(self) -> bool:
        return self._weekends

    @property
    def in_market(self) -> bool:
        return self._in_market

    @property
    def minutes_left(self) -> int:
        return self.__minute_to_market_close

    @property
    def last_updt(self) -> Optional[datetime]:
        return self._last_updt

    def update(self, current_time: int, passwd: bytes) -> None:
        if self.__passbyte != passwd:
            return

        dt_utc = datetime.fromtimestamp(current_time, tz=timezone.utc)
        self._last_updt = dt_utc

        if self._start_day <= dt_utc.time() <= self._end_day:
            if dt_utc.weekday() < 5: 
                self._in_market = True
            elif self._weekends:
                self._in_market = True
            else:
                self._in_market = False
        else:
            self._in_market = False

        if self._in_market:
            end_datetime = datetime.combine(dt_utc.date(), self._end_day, tzinfo=timezone.utc)
            time_left = end_datetime - dt_utc
            self.__minute_to_market_close = int(time_left.total_seconds() // 60)
        else:
            self.__minute_to_market_close = 0

    def __str__(self) -> str:
        schedule_type = "24/7" if self._weekends else "24/5"

        if self.in_market:
            status = f"Market running and {self.__minute_to_market_close} min to close"
        else:
            status = "Market off"

        last_update_str = (
            self._last_updt.strftime("%H:%M:%S UTC") if self._last_updt else "Never"
        )

        return f"{self._start_day}-{self._end_day} {schedule_type} | {status} | Last update: {last_update_str}"
