"""Mock for offline testing of Alicat devices."""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from random import choice, random
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

from .driver import CONTROL_POINTS, GASES, MAX_RAMP_TIME_UNITS, MaxRampTimeUnit
from .util import Client as RealClient

DeviceType = Literal['flow_meter', 'flow_controller', 'pressure_meter', 'pressure_controller']

# Keys for each device type, matching the driver classes
DEVICE_KEYS: dict[DeviceType, list[str]] = {
    'pressure_meter': ['pressure'],
    'pressure_controller': ['pressure', 'setpoint'],
    'flow_meter': ['pressure', 'temperature', 'volumetric_flow', 'mass_flow', 'gas'],
    'flow_controller': ['pressure', 'temperature', 'volumetric_flow', 'mass_flow', 'setpoint', 'gas'],
}

# Factory functions to generate default values for each state key
STATE_FACTORIES: dict[str, Callable[[], str | float]] = {
    'pressure': lambda: random() * 50.0,
    'temperature': lambda: random() * 50.0,
    'volumetric_flow': lambda: 0.0,
    'mass_flow': lambda: 10 * (0.95 + 0.1 * random()),
    'gas': lambda: 'N2',
    'setpoint': lambda: 10.0,
    'total flow': lambda: 0.0,  # Note: space to match driver key name
}


class Client(RealClient):
    """Mock the alicat communication client for all device types.

    Args:
        address: Serial port address (ignored in mock).
        device_type: Type of device to simulate:
            - 'flow_meter': FlowMeter (5 values: pressure, temp, vol_flow, mass_flow, gas)
            - 'flow_controller': FlowController (6 values: above + setpoint)
            - 'pressure_meter': PressureMeter (1 value: pressure)
            - 'pressure_controller': PressureController (2 values: pressure, setpoint)
    """

    def __init__(self, address: str, device_type: DeviceType = 'flow_controller') -> None:
        """Initialize mock client."""
        super().__init__(timeout=0.01)
        _ = address
        self.device_type = device_type
        self.keys = list(DEVICE_KEYS[device_type])  # copy to allow modification

        self.writer = MagicMock(spec=asyncio.StreamWriter)
        self.writer.write.side_effect = self._handle_write
        self.reader = AsyncMock(spec=asyncio.StreamReader)
        self.reader.read.return_value = self.eol
        self.reader.readuntil.side_effect = self._handle_read

        self.open = True
        self._next_reply = ''

        # Initialize state from keys - DEVICE_KEYS is the single source of truth
        self.state: dict[str, str | float] = {
            key: STATE_FACTORIES[key]() for key in self.keys
        }

        # Flow controllers have control_point
        if device_type == 'flow_controller':
            self.control_point: str = choice(list(CONTROL_POINTS))

        self.ramp_config = {'up': False, 'down': False, 'zero': False, 'power': False}
        self.button_lock: bool = False
        self.firmware = '6v21.0-R22 Nov 30 2016,16:04:20'
        self.max_ramp_time_unit: MaxRampTimeUnit = 's'
        self.max_ramp: float = 0.0

    async def _handle_connection(self) -> None:
        """Mock connection handler."""
        pass

    def _format_value(self, key: str) -> str:
        """Format a single value for the data frame."""
        value = self.state[key]
        if key == 'gas':
            return f"{value:<7}"
        elif key == 'setpoint':
            return f"{value:07.2f}"
        else:
            # Numeric values with sign
            return f"{value:+07.2f}"

    def _create_dataframe(self) -> str:
        """Generate a typical 'dataframe' with current operating conditions.

        Uses self.keys to determine which fields to include, matching
        how the driver classes work.
        """
        parts = [self.unit]
        for key in self.keys:
            parts.append(self._format_value(key))
        if self.button_lock:
            parts.append('LCK')
        return ' '.join(parts)

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
                f" {self.max_ramp:.7f}"
                f" 7"  # SLPM  # fixme make dynamic
                f" {MAX_RAMP_TIME_UNITS[self.max_ramp_time_unit]}"
                f" SLPM/{self.max_ramp_time_unit}")

    def _is_flow_device(self) -> bool:
        """Check if this is a flow device (meter or controller)."""
        return 'gas' in self.keys

    def _is_controller(self) -> bool:
        """Check if this is a controller (has setpoint)."""
        return 'setpoint' in self.keys

    def _handle_write(self, data: bytes) -> None:
        """Act on writes sent to the mock client, updating internal state."""
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
        elif msg == 'VE':  # get firmware
            self._next_reply = self.firmware
        elif msg == '$$PC':  # tare pressure
            self.state['pressure'] = 0
            self._next_reply = self._create_dataframe()
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
            self.max_ramp_time_unit = next(
                key for key, val in MAX_RAMP_TIME_UNITS.items() if val == unit_time_int
            )
            self._next_reply = self._create_max_ramp_response()
        elif msg[0] == 'S' and self._is_controller():  # set setpoint (controllers only)
            self.state['setpoint'] = float(msg[1:])
            self._next_reply = self._create_dataframe()
        # Flow controller only: control point commands
        elif self.device_type == 'flow_controller' and 'W122=' in msg:
            cp = int(msg[5:])
            self.control_point = next(p for p, i in CONTROL_POINTS.items() if cp == i)
            self._next_reply = f"{self.unit}   122 = {cp}"
        elif self.device_type == 'flow_controller' and msg == 'R122':
            self._next_reply = f"{self.unit}   122 = {CONTROL_POINTS[self.control_point]}"
        # Flow device (meter or controller): gas and flow commands
        elif self._is_flow_device() and msg[0:6] == '$$W46=':  # set gas via reg46
            gas = msg[6:]
            self._next_reply = f"{self.unit}   046 = {gas}"
            with contextlib.suppress(ValueError):
                gas = GASES[int(gas)]
            self.state['gas'] = gas
        elif self._is_flow_device() and msg == '$$R46':  # read gas via reg46
            gas_index = GASES.index(self.state['gas'])  # type: ignore[arg-type]
            reg46_value = gas_index & 0x1FF  # encode gas number in low 9 bits
            self._next_reply = f"{self.unit}   046 = {reg46_value}"
        elif self._is_flow_device() and msg == '$$V':  # tare volumetric flow
            self.state['volumetric_flow'] = 0
            self._next_reply = self._create_dataframe()
        else:
            raise NotImplementedError(msg)

    async def _handle_read(self, separator: bytes) -> bytes:
        """Reply to read requests from the mock client."""
        reply = self._next_reply.encode() + separator
        self._next_reply = ''
        return reply
