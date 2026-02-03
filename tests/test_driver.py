"""Test the driver responds with correct data."""
from random import uniform
from unittest import mock

import pytest

from alicat import command_line
from alicat.driver import AlicatDevice, FlowController, FlowMeter, PressureController, PressureMeter
from alicat.mock import Client

ADDRESS = '/dev/ttyUSB0'

# Mapping from device_type to driver class
DEVICE_CLASSES = {
    'flow_meter': FlowMeter,
    'flow_controller': FlowController,
    'pressure_meter': PressureMeter,
    'pressure_controller': PressureController,
}

FLOW_DEVICES = ['flow_meter', 'flow_controller']
PRESSURE_DEVICES = ['pressure_meter', 'pressure_controller']
CONTROLLERS = ['flow_controller', 'pressure_controller']
ALL_DEVICES = list(DEVICE_CLASSES.keys())


@pytest.fixture
def device(request):
    """Parameterized fixture that sets up mock client and yields device class.

    Usage:
        @pytest.mark.parametrize('device', ['flow_controller', 'pressure_meter'], indirect=True)
        async def test_something(device):
            async with device(ADDRESS) as d:
                ...
    """
    device_type = request.param
    AlicatDevice.open_ports.clear()

    def make_client(address, **kwargs):
        return Client(address, device_type=device_type, **kwargs)

    with mock.patch('alicat.driver.SerialClient', make_client):
        yield DEVICE_CLASSES[device_type]
    AlicatDevice.open_ports.clear()


# =============================================================================
# Tests for ALL devices
# =============================================================================

@pytest.mark.parametrize('device', ALL_DEVICES, indirect=True)
async def test_lock_unlock(device):
    """Confirm that locking/unlocking the buttons works on all devices."""
    async with device(ADDRESS) as d:
        await d.lock()
        assert await d.is_locked()
        await d.unlock()
        assert not await d.is_locked()


@pytest.mark.parametrize('device', ALL_DEVICES, indirect=True)
async def test_tare_pressure(device):
    """Confirm taring the pressure works on all devices."""
    async with device(ADDRESS) as d:
        await d.tare_pressure()
        result = await d.get()
        assert result['pressure'] == 0.0


@pytest.mark.parametrize('device', ALL_DEVICES, indirect=True)
async def test_get_firmware(device):
    """Confirm the firmware version can be read from all devices."""
    async with device(ADDRESS) as d:
        result = await d.get_firmware()
        assert 'v' in result or 'GP' in result


# =============================================================================
# Tests for FLOW devices (meter and controller)
# =============================================================================

@pytest.mark.parametrize('device', FLOW_DEVICES, indirect=True)
async def test_tare_flow(device):
    """Confirm taring the flow works on flow devices."""
    async with device(ADDRESS) as d:
        await d.tare_volumetric()
        result = await d.get()
        assert result['volumetric_flow'] == 0.0


@pytest.mark.parametrize('device', FLOW_DEVICES, indirect=True)
@pytest.mark.parametrize('gas', ['Air', 'H2'])
async def test_set_gas_by_name(device, gas):
    """Confirm that setting standard gases by name works on flow devices."""
    async with device(ADDRESS) as d:
        await d.set_gas(gas)
        result = await d.get()
        assert gas == result['gas']


@pytest.mark.parametrize('device', FLOW_DEVICES, indirect=True)
async def test_set_gas_invalid(device):
    """Confirm that setting an invalid gas raises an error."""
    async with device(ADDRESS) as d:
        with pytest.raises(ValueError, match='not supported'):
            await d.set_gas('methylacetylene-propadiene propane')


@pytest.mark.parametrize('device', FLOW_DEVICES, indirect=True)
@pytest.mark.parametrize('gas_name,gas_num', [('Air', 0), ('H2', 6)])
async def test_set_gas_by_number(device, gas_name, gas_num):
    """Confirm that setting standard gases by number works on flow devices."""
    async with device(ADDRESS) as d:
        await d.set_gas(gas_num)
        result = await d.get()
        assert gas_name == result['gas']


# =============================================================================
# Tests for CONTROLLERS (flow and pressure)
# =============================================================================

@pytest.mark.parametrize('device', CONTROLLERS, indirect=True)
@pytest.mark.parametrize('config', [
    {'up': True, 'down': False, 'zero': True, 'power': False},
    {'up': True, 'down': True, 'zero': False, 'power': True},
    {'up': False, 'down': False, 'zero': False, 'power': False},
])
async def test_ramp_config(device, config):
    """Confirm changing the ramping configuration works on controllers."""
    async with device(ADDRESS) as d:
        await d.set_ramp_config(config)
        result = await d.get_ramp_config()
        assert config == result


@pytest.mark.parametrize('device', CONTROLLERS, indirect=True)
@pytest.mark.parametrize('unit_time', ['ms', 's', 'm', 'h', 'd'])
async def test_maxramp(device, unit_time):
    """Confirm that setting/getting the maximum ramp rate works on controllers."""
    async with device(ADDRESS) as d:
        max_ramp = round(uniform(0.01, 0.1), 2)
        await d.set_maxramp(max_ramp, unit_time)
        result = await d.get_maxramp()
        assert max_ramp == result['max_ramp']
        assert result['units'] == f'SLPM/{unit_time}'  # fixme make units dynamic


# =============================================================================
# Tests specific to FlowController
# =============================================================================

@pytest.mark.parametrize('device', ['flow_controller'], indirect=True)
async def test_flow_controller_get_fields(device):
    """Confirm that FlowController.get() returns expected keys."""
    async with device(ADDRESS) as d:
        result = await d.get()
        assert 'pressure' in result
        assert 'temperature' in result
        assert 'volumetric_flow' in result
        assert 'mass_flow' in result
        assert 'setpoint' in result
        assert 'gas' in result
        assert 'control_point' in result
        assert 'status' in result
        # 6 data fields + control_point + status = 8
        assert len(result) == 8


@pytest.mark.parametrize('device', ['flow_controller'], indirect=True)
async def test_flow_controller_is_connected(device):
    """Confirm that connection status works for flow controller."""
    async with device(ADDRESS) as d:
        assert await d.is_connected(ADDRESS)
        assert not await d.is_connected('bad_address')


@pytest.mark.parametrize('device', ['flow_controller'], indirect=True)
async def test_flow_setpoint_roundtrip(device):
    """Confirm that setting/getting flowrates works."""
    async with device(ADDRESS) as d:
        flow_sp = round(uniform(0.01, 0.1), 2)
        await d.set_flow_rate(flowrate=flow_sp)
        result = await d.get()
        assert flow_sp == result['setpoint']


@pytest.mark.parametrize('device', ['flow_controller'], indirect=True)
@pytest.mark.parametrize('control_point',
    ['mass flow', 'vol flow', 'abs pressure', 'gauge pressure', 'diff pressure'])
async def test_control_point(device, control_point):
    """Confirm changing the control point works."""
    async with device(ADDRESS) as d:
        await d._set_control_point(control_point)
        result = await d._get_control_point()
        assert control_point == result


@pytest.mark.parametrize('device', ['flow_controller'], indirect=True)
@pytest.mark.parametrize('unit', ['A', 'B'])
def test_driver_cli(device, capsys, unit):
    """Confirm the commandline interface works with different unit IDs."""
    command_line([ADDRESS, '--unit', unit])
    captured = capsys.readouterr()
    assert ("mass_flow" in captured.out)


@pytest.mark.skip
async def test_create_gas_mix():
    """Confirm creating custom gas mixes works."""
    pass


# =============================================================================
# Tests specific to FlowMeter
# =============================================================================

@pytest.mark.parametrize('device', ['flow_meter'], indirect=True)
async def test_flow_meter_get_fields(device):
    """Confirm that FlowMeter.get() returns 5 data fields + status (no setpoint)."""
    async with device(ADDRESS) as d:
        result = await d.get()
        assert 'pressure' in result
        assert 'temperature' in result
        assert 'volumetric_flow' in result
        assert 'mass_flow' in result
        assert 'gas' in result
        assert 'status' in result
        assert 'setpoint' not in result
        # 5 data fields + status = 6
        assert len(result) == 6


# =============================================================================
# Tests specific to PressureController
# =============================================================================

@pytest.mark.parametrize('device', ['pressure_controller'], indirect=True)
async def test_pressure_controller_get_fields(device):
    """Confirm that PressureController.get() returns pressure, setpoint, and status."""
    async with device(ADDRESS) as d:
        result = await d.get()
        assert 'pressure' in result
        assert 'setpoint' in result
        assert 'status' in result
        # 2 data fields + status = 3
        assert len(result) == 3


@pytest.mark.parametrize('device', ['pressure_controller'], indirect=True)
async def test_pressure_controller_set_pressure(device):
    """Confirm that setting pressure works."""
    async with device(ADDRESS) as d:
        pressure_sp = round(uniform(10.0, 20.0), 2)
        await d.set_pressure(pressure_sp)
        result = await d.get()
        assert result['setpoint'] == pressure_sp


# =============================================================================
# Tests specific to PressureMeter
# =============================================================================

@pytest.mark.parametrize('device', ['pressure_meter'], indirect=True)
async def test_pressure_meter_get_fields(device):
    """Confirm that PressureMeter.get() returns only pressure and status."""
    async with device(ADDRESS) as d:
        result = await d.get()
        assert 'pressure' in result
        assert 'status' in result
        # 1 data field + status = 2
        assert len(result) == 2
