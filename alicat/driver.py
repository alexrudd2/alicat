"""Python driver for Alicat mass flow controllers, using serial communication.

Distributed under the GNU General Public License v2
Copyright (C) 2024 Alex Ruddick
Copyright (C) 2023 NuMat Technologies
"""
from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Literal, TypeVar

from .util import Client, SerialClient, TcpClient, _is_float

_T = TypeVar('_T', bound='AlicatDevice')

GASES = ['Air', 'Ar', 'CH4', 'CO', 'CO2', 'C2H6', 'H2', 'He',
         'N2', 'N2O', 'Ne', 'O2', 'C3H8', 'n-C4H10', 'C2H2',
         'C2H4', 'i-C2H10', 'Kr', 'Xe', 'SF6', 'C-25', 'C-10',
         'C-8', 'C-2', 'C-75', 'A-75', 'A-25', 'A1025', 'Star29',
         'P-5']

# Status flags that can appear at the end of device responses.
# Reference: Alicat Serial Primer, page 8 (https://documents.alicat.com/Alicat-Serial-Primer.pdf)
STATUS_FLAGS = frozenset({
    'EXH',  # exhaust valve override enabled
    'HLD',  # valve hold enabled
    'LCK',  # display buttons locked
    'MOV',  # mass flow overage
    'OPL',  # overpressure limit enabled
    'OVR',  # totalizer rollover
    'POV',  # pressure overage
    'TMF',  # totalizer missed flow
    'TOV',  # temperature overage
    'VOV',  # volumetric flow overage
})
# Error flags that indicate hardware failures and should raise exceptions immediately
ERROR_FLAGS = frozenset({
    'ADC',  # internal communication error
})

MaxRampTimeUnit = Literal['ms', 's', 'm', 'h', 'd']
MAX_RAMP_TIME_UNITS: dict[MaxRampTimeUnit, int] = {
    'ms': 3,
    's': 4,
    'm': 5,
    'h': 6,
    'd': 7,
}


class AlicatDevice:
    """Base class for all Alicat devices.

    This communicates with the flow meter over a USB or RS-232/RS-485
    connection using pyserial, or an Ethernet <-> serial converter.
    Provides universal commands shared by all device types (lock, unlock,
    firmware, etc.).
    """

    # mapping of port names to a tuple of Client objects and their refcounts
    open_ports: ClassVar[dict[str, tuple[Client, int]]] = {}

    def __init__(self, address: str = '/dev/ttyUSB0', unit: str = 'A', **kwargs: Any) -> None:
        """Connect this driver with the appropriate USB / serial port.

        Args:
            address: The serial port or TCP address:port. Default '/dev/ttyUSB0'.
            unit: The Alicat-specified unit ID, A-Z. Default 'A'.
        """
        if address.startswith('/dev') or address.startswith('COM'):  # serial
            if address in AlicatDevice.open_ports:
                # Reuse existing connection
                self.hw, refcount = AlicatDevice.open_ports[address]
                AlicatDevice.open_ports[address] = (self.hw, refcount + 1)
            else:
                # Open a new connection and store it
                self.hw: Client = SerialClient(address=address, **kwargs)  # type: ignore[no-redef]
                AlicatDevice.open_ports[address] = (self.hw, 1)
        else:
            self.hw = TcpClient(address=address, **kwargs)

        self.unit = unit
        self.open = True
        self.firmware: str | None = None
        self.keys: list[str]  # set by subclasses

    async def __aenter__(self: _T) -> _T:
        """Provide async enter to context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Provide async exit to context manager."""
        await self.close()
        return

    def _test_controller_open(self) -> None:
        """Raise an IOError if the device has been closed.

        Does nothing if the device is open and good for read/write
        otherwise raises an IOError. This only checks if the device
        has been closed by the close method.
        """
        if not self.open:
            raise OSError(f"The device with unit ID {self.unit} and "
                          f"port {self.hw.address} is not open")

    async def _write_and_read(self, command: str) -> str | None:
        """Wrap the communicator request, to call _test_controller_open() before any request."""
        self._test_controller_open()
        return await self.hw.write_and_read(command)

    def _parse_response(self, line: str) -> tuple[list[str], set[str]]:
        """Parse a device response, extracting data values and status flags.

        Args:
            line: The raw response line from the device

        Returns:
            A tuple of (values, flags) where values is the list of data fields
            (excluding unit ID and flags) and flags is a set of status flags found.

        Raises:
            OSError: If ADC (internal communication error) flag is present
            ValueError: If the unit ID in the response doesn't match
        """
        spl = line.split()
        unit = spl[0]
        values = spl[1:]

        if unit != self.unit:
            raise ValueError("Device unit ID mismatch.")

        # Extract flags from the end of the response
        flags: set[str] = set()
        while values and values[-1].upper() in (STATUS_FLAGS | ERROR_FLAGS):
            flag = values.pop().upper()
            if flag in ERROR_FLAGS:
                raise OSError(f"Device error: {flag}")
            flags.add(flag)

        return values, flags

    async def lock(self) -> None:
        """Lock the buttons."""
        command = f'{self.unit}$$L'
        response = await self._write_and_read(command)
        if not response or 'LCK' not in response.upper():
            raise OSError("Failed to lock device buttons.")

    async def unlock(self) -> None:
        """Unlock the buttons."""
        command = f'{self.unit}$$U'
        response = await self._write_and_read(command)
        if not response or 'LCK' in response.upper():
            raise OSError("Failed to unlock device buttons.")

    async def is_locked(self) -> bool:
        """Return whether the buttons are locked."""
        command = f'{self.unit}'
        response = await self._write_and_read(command)
        return response is not None and 'LCK' in response.upper()

    async def get(self) -> dict[str, Any]:
        """Get the current state of the device.

        Polls the device and returns a dictionary of values based on self.keys,
        which must be set by subclasses.

        Returns:
            The state of the device, as a dictionary. Includes a 'status'
            key containing a sorted list of any status flags present in the response.
        """
        command = f'{self.unit}'
        line = await self._write_and_read(command)
        if not line:
            raise OSError("Could not read values")

        values, flags = self._parse_response(line)

        if len(values) != len(self.keys):
            raise ValueError(
                f"Response field count mismatch: got {len(values)} values "
                f"but expected {len(self.keys)} (keys: {self.keys}). "
                f"Check device type and totalizer setting.",
            )

        result: dict[str, Any] = {k: (float(v) if _is_float(v) else v)
                                  for k, v in zip(self.keys, values, strict=True)}
        result['status'] = sorted(flags)
        return result

    async def tare_pressure(self) -> None:
        """Tare the pressure."""
        command = f'{self.unit}$$PC'
        line = await self._write_and_read(command)
        if line == '?':
            raise OSError("Unable to tare pressure.")

    async def get_firmware(self) -> str:
        """Get the device firmware version."""
        if self.firmware is None:
            command = f'{self.unit}VE'
            self.firmware = await self._write_and_read(command)
        if not self.firmware:
            raise OSError("Unable to get firmware.")
        return self.firmware

    async def flush(self) -> None:
        """Read all available information. Use to clear queue."""
        self._test_controller_open()
        await self.hw.clear()

    async def close(self) -> None:
        """Close the device. Call this on program termination.

        Also close the serial port if no other device object has
        a reference to the port.
        """
        if not self.open:
            return
        port = self.hw.address
        if port in AlicatDevice.open_ports:
            connection, refcount = AlicatDevice.open_ports[port]
            if refcount > 1:
                AlicatDevice.open_ports[port] = (connection, refcount - 1)
            else:
                await connection.close()  # Close the port if no other instance uses it
                del AlicatDevice.open_ports[port]
        self.open = False

    @classmethod
    async def is_connected(cls, port: str, unit: str = 'A',
                           totalizer: bool = False) -> bool:
        """Check whether the specified port has a device of this type.

        Args:
            port: The serial port or TCP address:port.
            unit: The Alicat-specified unit ID, A-Z. Default 'A'.
            totalizer: Whether the device has totalizer enabled. Default False.

        Returns:
            True if a device responds with the expected number of fields.

        Note: This method can produce false positives. A FlowMeter with totalizer
        and a FlowController without totalizer both return 6 fields, so this
        method cannot distinguish between them based on field count alone.
        """
        if cls is AlicatDevice:
            raise NotImplementedError("Call is_connected on a subclass, not AlicatDevice")
        try:
            device = cls(port, unit, totalizer=totalizer)
            try:
                await device.get()
                return True
            finally:
                await device.close()
        except Exception:
            return False


class FlowMeter(AlicatDevice):
    """Python driver for Alicat Flow Meters.

    [Reference](http://www.alicat.com/
    products/mass-flow-meters-and-controllers/mass-flow-meters/).

    This communicates with the flow meter over a USB or RS-232/RS-485
    connection using pyserial, or an Ethernet <-> serial converter.
    """

    # Keep class-level open_ports for backwards compatibility
    # (some code may reference FlowMeter.open_ports directly)
    open_ports: ClassVar[dict[str, tuple[Client, int]]] = AlicatDevice.open_ports

    def __init__(self, address: str = '/dev/ttyUSB0', unit: str = 'A',
                 totalizer: bool = False, **kwargs: Any) -> None:
        """Connect this driver with the appropriate USB / serial port.

        Args:
            address: The serial port or TCP address:port. Default '/dev/ttyUSB0'.
            unit: The Alicat-specified unit ID, A-Z. Default 'A'.
            totalizer: Whether the device has totalizer enabled. Default False.
        """
        super().__init__(address, unit, **kwargs)
        # FlowMeter: no setpoint field (meter only)
        self.keys = ['pressure', 'temperature', 'volumetric_flow', 'mass_flow', 'gas']
        if totalizer:
            self.keys.insert(4, 'total flow')

    async def set_gas(self, gas: str | int) -> None:
        """Set the gas type.

        Args:
            gas: The gas type, as a string or integer. Supported strings are:
                'Air', 'Ar', 'CH4', 'CO', 'CO2', 'C2H6', 'H2', 'He', 'N2',
                'N2O', 'Ne', 'O2', 'C3H8', 'n-C4H10', 'C2H2', 'C2H4',
                'i-C2H10', 'Kr', 'Xe', 'SF6', 'C-25', 'C-10', 'C-8', 'C-2',
                'C-75', 'A-75', 'A-25', 'A1025', 'Star29', 'P-5'

                Gas mixes may only be called by their mix number.
        """
        if isinstance(gas, str):
            if gas not in GASES:
                raise ValueError(f"{gas} not supported!")
            gas_number = GASES.index(gas)
        else:
            gas_number = gas
        command = f'{self.unit}$$W46={gas_number}'
        # fixme does this overwrite the upper bits??
        _ = await self._write_and_read(command)
        reg46 = await self._write_and_read(f'{self.unit}$$R46')
        if not reg46:
            raise OSError("Cannot set gas.")
        reg46_gasbit = int(reg46.split()[-1]) & 0b0000000111111111

        if gas_number != reg46_gasbit:
            raise OSError("Cannot set gas.")

    async def create_mix(self, mix_no: int, name: str, gases: dict[str, float]) -> None:
        """Create a gas mix.

        Gas mixes are made using COMPOSER software.
        COMPOSER mixes are only allowed for firmware 5v or greater.

        Args:
        mix_no: The mix number. Gas mixes are stored in slots 236-255.
        name: A name for the gas that will appear on the front panel.
        Names greater than six letters will be cut off.
        gases: A dictionary of the gas by name along with the associated
        percentage in the mix.
        """
        firmware = await self.get_firmware()
        if any(v in firmware for v in ['2v', '3v', '4v', 'GP']):
            raise OSError("This unit does not support COMPOSER gas mixes.")

        if mix_no < 236 or mix_no > 255:
            raise ValueError("Mix number must be between 236-255!")

        total_percent = sum(gases.values())
        if total_percent != 100:
            raise ValueError("Percentages of gas mix must add to 100%!")

        if any(gas not in GASES for gas in gases):
            raise ValueError("Gas not supported!")

        gas_list = [f'{percent} {GASES.index(gas)}' for gas, percent in gases.items()]
        command = ' '.join([
            self.unit,
            'GM',
            name,
            str(mix_no),
            ' '.join(gas_list),
        ])
        line = await self._write_and_read(command)

        # If a gas mix is not successfully created, ? is returned.
        if line == '?':
            raise OSError("Unable to create mix.")

    async def delete_mix(self, mix_no: int) -> None:
        """Delete a gas mix."""
        command = f'{self.unit}GD{mix_no}'
        line = await self._write_and_read(command)
        if line == '?':
            raise OSError("Unable to delete mix.")

    async def tare_volumetric(self) -> None:
        """Tare volumetric flow."""
        command = f'{self.unit}$$V'
        line = await self._write_and_read(command)
        if line == '?':
            raise OSError("Unable to tare flow.")

    async def reset_totalizer(self) -> None:
        """Reset the totalizer."""
        command = f'{self.unit}T'
        _ = await self._write_and_read(command)


CONTROL_POINTS = {
    'mass flow': 37, 'vol flow': 36,
    'abs pressure': 34, 'gauge pressure': 38, 'diff pressure': 39,
}  # fixme: add remaining control points


class ControllerMixin:
    """Mixin providing controller functionality (PID, hold, ramp, setpoint)."""

    unit: str  # provided by AlicatDevice
    keys: list[str]  # provided by FlowMeter/subclasses

    async def _write_and_read(self, command: str) -> str | None:
        """Write command and return response. Provided by AlicatDevice."""
        raise NotImplementedError

    async def _set_setpoint(self, setpoint: float) -> None:
        """Set the target setpoint.

        Called by `set_flow_rate` and `set_pressure`, which both use the same
        command once the appropriate register is set.
        """
        command = f'{self.unit}S{setpoint:.2f}'
        line = await self._write_and_read(command)
        if not line:
            raise OSError("Could not set setpoint.")
        # Response format: {unit_id} {fields...} where fields follow self.keys
        # Add 1 to account for unit_id prefix in response
        setpoint_index = self.keys.index('setpoint') + 1
        current = float(line.split()[setpoint_index])
        if abs(current - setpoint) > 0.01:
            # possibly the setpoint is being ramped
            command = f'{self.unit}LS'
            line = await self._write_and_read(command)
            if not line:
                raise OSError("Could not set setpoint.")
            commanded = float(line.split()[2])
            if abs(commanded - setpoint) > 0.01:
                raise OSError("Could not set setpoint.")

    async def hold(self) -> None:
        """Override command to issue a valve hold (firmware 5v07).

        For a single valve controller, hold the valve at the present value.
        For a dual valve flow controller, hold the valve at the present value.
        For a dual valve pressure controller, close both valves.
        """
        command = f'{self.unit}$$H'
        _ = await self._write_and_read(command)

    async def cancel_hold(self) -> None:
        """Cancel valve hold."""
        command = f'{self.unit}$$C'
        _ = await self._write_and_read(command)

    async def get_pid(self) -> dict[str, Any]:
        """Read the current PID values on the controller.

        Values include the loop type, P value, D value, and I value.
        Values returned as a dictionary.
        """
        pid_keys = ['loop_type', 'P', 'D', 'I']

        command = f'{self.unit}$$r85'
        read_loop_type = await self._write_and_read(command)
        if not read_loop_type:
            raise OSError("Could not get PID values.")
        spl = read_loop_type.split()

        loopnum = int(spl[3])
        loop_type = ['PD/PDF', 'PD/PDF', 'PD2I'][loopnum]
        pid_values: list[Any] = [loop_type]
        for register in range(21, 24):
            value = await self._write_and_read(f'{self.unit}$$r{register}')
            if not value:
                raise OSError(f"Could not read register {register}")
            value_spl = value.split()
            pid_values.append(value_spl[3])

        return {k: (v if k == pid_keys[-1] else str(v))
                for k, v in zip(pid_keys, pid_values, strict=True)}

    async def set_pid(self, p: int | None = None,
                      i: int | None = None,
                      d: int | None = None,
                      loop_type: str | None = None) -> None:
        """Set specified PID parameters.

        Args:
            p: Proportional gain
            i: Integral gain. Only used in PD2I loop type.
            d: Derivative gain
            loop_type: Algorithm option, either 'PD/PDF' or 'PD2I'

        This communication works by writing Alicat registers directly.
        """
        if loop_type is not None:
            options = ['PD/PDF', 'PD2I']
            if loop_type not in options:
                raise ValueError(f'Loop type must be {options[0]} or {options[1]}.')
            loop_num = options.index(loop_type) + 1
            command = f'{self.unit}$$w85={loop_num}'
            _ = await self._write_and_read(command)
        if p is not None:
            command = f'{self.unit}$$w21={p}'
            _ = await self._write_and_read(command)
        if i is not None:
            command = f'{self.unit}$$w23={i}'
            _ = await self._write_and_read(command)
        if d is not None:
            command = f'{self.unit}$$w22={d}'
            _ = await self._write_and_read(command)

    async def set_ramp_config(self, config: dict[str, bool]) -> None:
        """Configure the setpoint ramp settings (firmware 10v05).

        Args:
            config: Dictionary with boolean keys:
                `up`: whether to ramp when increasing setpoint
                `down`: whether to ramp when decreasing setpoint
                `zero`: whether to ramp when establishing zero setpoint
                `power`: whether to ramp when using power-up setpoint
        """
        command = (f"{self.unit}LSRC"
                   f" {1 if config['up'] else 0}"
                   f" {1 if config['down'] else 0}"
                   f" {1 if config['zero'] else 0}"
                   f" {1 if config['power'] else 0}")
        line = await self._write_and_read(command)
        if not line or self.unit not in line:
            raise OSError("Could not set ramp config.")

    async def get_ramp_config(self) -> dict[str, bool]:
        """Get the setpoint ramp settings (firmware 10v05).

        Returns:
            Dictionary with boolean keys: up, down, zero, power
        """
        command = f"{self.unit}LSRC"
        line = await self._write_and_read(command)
        if not line or self.unit not in line:
            raise OSError("Could not read ramp config.")
        values = line[2:].split(' ')
        if len(values) != 4:
            raise OSError("Could not read ramp config.")
        return {
            'up': values[0] == '1',
            'down': values[1] == '1',
            'zero': values[2] == '1',
            'power': values[3] == '1',
        }

    async def set_maxramp(self, max_ramp: float,
                          unit_time: MaxRampTimeUnit) -> None:
        """Set the maximum ramp rate (firmware 7v11).

        Args:
            max_ramp: The maximum ramp rate
            unit_time: The time unit ('ms', 's', 'm', 'h', 'd')
        """
        command = f"{self.unit}SR {max_ramp:.2f} {MAX_RAMP_TIME_UNITS[unit_time]}"
        line = await self._write_and_read(command)
        if not line or self.unit not in line:
            raise OSError("Could not set max ramp.")

    async def get_maxramp(self) -> dict[str, float | str]:
        """Get the maximum ramp rate (firmware 7v11).

        Returns:
            Dictionary with 'max_ramp' (float) and 'units' (str)
        """
        command = f"{self.unit}SR"
        line = await self._write_and_read(command)
        if not line or self.unit not in line:
            raise OSError("Could not read max ramp.")
        values = line.split(' ')
        if len(values) != 5:
            raise OSError("Could not read max ramp.")
        return {
            'max_ramp': float(values[1]),
            'units': str(values[4]),
        }


class FlowController(FlowMeter, ControllerMixin):
    """Python driver for Alicat Flow Controllers.

    [Reference](http://www.alicat.com/products/mass-flow-meters-and-
    controllers/mass-flow-controllers/).

    This communicates with the flow controller over a USB or RS-232/RS-485
    connection using pyserial.

    To set up your Alicat flow controller, power on the device and make sure
    that the "Input" option is set to "Serial".
    """

    def __init__(self, address: str = '/dev/ttyUSB0', unit: str = 'A',
                 totalizer: bool = False, **kwargs: Any) -> None:
        """Connect this driver with the appropriate USB / serial port.

        Args:
            address: The serial port or TCP address:port. Default '/dev/ttyUSB0'.
            unit: The Alicat-specified unit ID, A-Z. Default 'A'.
            totalizer: Whether the device has totalizer enabled. Default False.
        """
        FlowMeter.__init__(self, address, unit, totalizer=totalizer, **kwargs)
        # FlowController: has setpoint field (insert before gas)
        self.keys.insert(self.keys.index('gas'), 'setpoint')
        self.control_point: str | None = None

        async def _init_control_point() -> None:
            self.control_point = await self._get_control_point(_init=True)
        self._init_task = asyncio.create_task(_init_control_point())

    async def _write_and_read(self, command: str, *, _init: bool = False) -> str | None:
        """Ensure _init_task completes before any command except init commands."""
        if not _init:
            await self._init_task
        return await super()._write_and_read(command)

    async def get(self) -> dict[str, Any]:
        """Get the current state of the flow controller.

        Overrides base to add 'control_point', which is not part of the
        device's data frame but is cached by the driver from register 122.
        """
        state = await super().get()
        state['control_point'] = self.control_point
        return state

    async def set_flow_rate(self, flowrate: float) -> None:
        """Set the target flow rate.

        Args:
            flow: The target flow rate, in units specified at time of purchase
        """
        if self.control_point in ['abs pressure', 'gauge pressure', 'diff pressure']:
            await self._set_setpoint(0)
            await self._set_control_point('mass flow')
        await self._set_setpoint(flowrate)

    async def set_pressure(self, pressure: float) -> None:
        """Set the target pressure.

        Args:
            pressure: The target pressure, in units specified at time of
                purchase. Likely in psia.
        """
        if self.control_point in ['mass flow', 'vol flow']:
            await self._set_setpoint(0)
            await self._set_control_point('abs pressure')
        await self._set_setpoint(pressure)

    async def get_totalizer_batch(self, batch: int = 1) -> str:
        """Get the totalizer batch volume (firmware 10v00).

        Args:
            batch: Which of the two totalizer batches to query.
                Default is 1; some devices have 2

        Returns:
            line: Current value of totalizer batch
        """
        command = f'{self.unit}$$TB {batch}'
        line = await self._write_and_read(command)
        if line == '?':
            raise OSError("Unable to read totalizer batch volume.")
        values = line.split(" ")  # type: ignore[union-attr]
        return f'{values[2]} {values[4]}' # returns 'batch vol' 'units'

    async def set_totalizer_batch(self, batch_volume: float, batch: int = 1, units: str = 'default') -> None:
        """Set the totalizer batch volume (firmware 10v00).

        Args:
            batch: Which of the two totalizer batches to set.
                Default is 1; some devices have 2
            batch_volume: Target batch volume, in same units as units
                on device
            units: Units of the volume being provided. Default
                is 0, so device returns default engineering units.
        """
        engineering_units_table = {"default":0, "SμL":2, "SmL":3, "SL":4, \
                    "Scm3":6, "Sm3":7, "Sin3":8, "Sft3":9, "kSft3":10, "NμL":32, \
                    "NmL":33, "NL":34, "Ncm3":36, "Nm3":37}

        if units in engineering_units_table:
            units_no = engineering_units_table[units]
        else:
            raise ValueError("Units not in unit list. Please consult Appendix B-3 of the Alicat Serial Primer.")

        command = f'{self.unit}$$TB {batch} {batch_volume} {units_no}'
        line = await self._write_and_read(command)
        if line == '?':
            raise OSError("Unable to set totalizer batch volume. Check if volume is out of range for device.")

    async def _get_control_point(self, *, _init: bool = False) -> str:
        """Get the control point, and save to internal variable."""
        command = f'{self.unit}R122'
        line = await self._write_and_read(command, _init=_init)
        if not line:
            raise OSError("Could not read control point.")
        value = int(line.split('=')[-1])
        try:
            cp = next(p for p, r in CONTROL_POINTS.items() if value == r)
            self.control_point = cp
            return cp
        except StopIteration:
            raise ValueError(f"Unexpected register value: {value:d}") from None

    async def _set_control_point(self, point: str) -> None:
        """Set the variable used as the control point.

        Args:
            point: 'mass flow', 'vol flow', 'abs pressure', 'gauge pressure', or 'diff pressure'
        """
        if point not in CONTROL_POINTS:
            raise ValueError(f"Control point must be one of {list(CONTROL_POINTS.keys())}.")
        reg = CONTROL_POINTS[point]
        command = f'{self.unit}W122={reg:d}'
        line = await self._write_and_read(command)
        if not line:
            raise OSError("Could not set control point.")
        value = int(line.split('=')[-1])
        if value != reg:
            raise OSError("Could not set control point.")
        self.control_point = point
