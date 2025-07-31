"""Mock for offline testing of `FlowController`s."""
from __future__ import annotations

import asyncio
import contextlib
from random import choice, random
from unittest.mock import AsyncMock, MagicMock

from .driver import CONTROL_POINTS, GASES, MAX_RAMP_TIME_UNITS
from .util import Client as RealClient


class Client(RealClient):
    """Mock the alicat communication client."""

    def __init__(self, address: str) -> None:
        super().__init__(timeout=0.01)
        self.writer = MagicMock(spec=asyncio.StreamWriter)
        self.writer.write.side_effect=self._handle_write
        self.reader = AsyncMock(spec=asyncio.StreamReader)
        self.reader.read.return_value = self.eol
        self.reader.readuntil.side_effect = self._handle_read

        self.open = True
        self._next_reply = ''

        self.open = True
        self.control_point: str = choice(list(CONTROL_POINTS))  # type:ignore[assignment]
        self.state: dict[str, str | float] = {
            'setpoint': 10,
            'gas': 'N2',
            'mass_flow': 10 * (0.95 + 0.1 * random()),
            'pressure': random() * 50.0,
            'temperature': random() * 50.0,
            'total_flow': 0.0,
            'volumetric_flow': 0.0,
        }
        self.ramp_config = { 'up': False, 'down': False, 'zero': False, 'power': False }
        self.button_lock: bool = False
        self.keys = ['pressure', 'temperature', 'volumetric_flow', 'mass_flow',
                     'setpoint', 'gas']
        self.firmware = '6v21.0-R22 Nov 30 2016,16:04:20'

    async def _handle_connection(self) -> None:
        pass

    def _create_dataframe(self) -> str:
        """Generate a typical 'dataframe' with current operating conditions."""
        return (
            f"{self.unit} "
            f"{self.state['pressure']:+07.2f} "
            f"{self.state['temperature']:+07.2f} "
            f"{self.state['volumetric_flow']:+07.2f} "
            f"{self.state['mass_flow']:+07.2f} "
            f"{self.state['setpoint']:07.2f} "
            f"{self.state['gas']:<7} "
            f"{'LCK' if self.button_lock else ''}"
        )
    def _create_ramp_response(self) -> str:
        """Generate a response to setting or getting the ramp config."""
        config = self.ramp_config
        return (f"{self.unit}"
                f" {1 if config['up'] else 0}"
                f" {1 if config['down'] else 0}"
                f" {1 if config['zero'] else 0}"
                f" {1 if config['power'] else 0}")

    def _create_max_ramp_response(self) -> str:
        """Generate a response to setting or getting the max ramp rate."""
        return (f"{self.unit}"
                f" {self.max_ramp:+07.2f}"
                f" 37"  # SLPM  # fixme make dynamic
                f" {MAX_RAMP_TIME_UNITS[self.max_ramp_time_unit]}"
                f" SLPM/s")

    def _handle_write(self, data: bytes) -> None:
        """Act on writes sent to the mock client, updating internal state and setting self._next_reply if necessary."""
        msg = data.decode().strip()
        self.unit = msg[0]

        msg = msg[1:]  # strip unit
        if msg == '':  # get dataframe
            self._next_reply = self._create_dataframe()
        elif msg == '$$L':  # lock
            self.button_lock = True
            self._next_reply = self._create_dataframe()
        elif msg == '$$U':  # unlock
            self.button_lock = False
            self._next_reply = self._create_dataframe()
        elif 'W122=' in msg:  # set control point
            cp = int(msg[5:])
            self.control_point = next(p for p, i in CONTROL_POINTS.items() if cp == i)
            self._next_reply = f"{self.unit}   122 = {cp}"
        elif msg == 'R122':  # read control point
            self._next_reply = f"{self.unit}   122 = {CONTROL_POINTS[self.control_point]}"
        elif msg == 'LSRC':  # get ramp config
            self._next_reply = self._create_ramp_response()
        elif 'LSRC' in msg:  # set ramp config
            values = msg[5:].split(' ')
            self.ramp_config = {
                'up': values[0] == '1',
                'down': values[1] == '1',
                'zero': values[2] == '1',
                'power': values[3] == '1',
            }
            self._next_reply = self._create_ramp_response()
        elif msg == 'SR':  # get max ramp rate
            self._next_reply = self._create_max_ramp_response()
        elif 'SR' in msg:  # set max ramp rate
            values = msg.split()
            self.max_ramp = float(values[1])
            unit_time_int = int(values[2])
            self.max_ramp_time_unit = next(key for key, val in MAX_RAMP_TIME_UNITS.items() if val == unit_time_int)
            self._next_reply = self._create_max_ramp_response()
        elif msg[0] == 'S':  # set setpoint
            self.state['setpoint'] = float(msg[1:])
            self._next_reply = self._create_dataframe()
        elif msg[0:6] == '$$W46=':  # set gas via reg46
            gas = msg[6:]
            self._next_reply = f"{self.unit}   046 = {gas}"
            with contextlib.suppress(ValueError):
                gas = GASES[int(gas)]
            self.state['gas'] = gas
        elif msg == '$$R46':  # read gas via reg46
            gas_index = GASES.index(self.state['gas'])  # type: ignore
            reg46_value = gas_index & 0x1FF  # encode gas number in low 9 bits
            self._next_reply = f"{self.unit}   046 = {reg46_value}"
        elif msg == 'VE':  # get firmware
            self._next_reply = self.firmware
        else:
            raise NotImplementedError(msg)

    async def _handle_read(self, separator: bytes) -> bytes:
        """Reply to read requests from the mock client."""
        reply = self._next_reply.encode() + separator
        self._next_reply = ''
        return reply
