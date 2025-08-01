"""Test the driver responds with correct data."""
from random import uniform
from unittest import mock

import pytest

from alicat import command_line
from alicat.driver import FlowController
from alicat.mock import Client

ADDRESS = '/dev/ttyUSB0'

@mock.patch('alicat.driver.SerialClient', Client)

@pytest.mark.parametrize('unit', ['A', 'B'])
def test_driver_cli(capsys, unit):
    """Confirm the commandline interface works with different unit IDs."""
    command_line([ADDRESS, '--unit', unit])
    captured = capsys.readouterr()
    assert ("mass_flow" in captured.out)


async def test_flow_setpoint_roundtrip():
    """Confirm that setting/getting flowrates works."""
    async with FlowController(ADDRESS) as device:
        flow_sp = round(uniform(0.01, 0.1), 2)
        await device.set_flow_rate(flowrate=flow_sp)
        # assert flow_sp == await device.get_flow_rate()
        result = await device.get()
        assert flow_sp == result['setpoint']


async def test_lock_unlock():
    """Confirm that locking/unlocking the buttons works."""
    async with FlowController(ADDRESS) as device:
        await device.lock()
        assert await device.is_locked()
        await device.unlock()
        assert not await device.is_locked()


@pytest.mark.parametrize('gas', ['Air', 'H2'])
async def test_set_standard_gas_name(gas):
    """Confirm that setting standard gases by name works."""
    async with FlowController(ADDRESS) as device:
        await device.set_gas(gas)
        result = await device.get()
        assert gas == result['gas']


@pytest.mark.parametrize('gas', [('Air', 0), ('H2', 6)])
async def test_set_standard_gas_number(gas):
    """Confirm that setting standard gases by number works."""
    async with FlowController(ADDRESS) as device:
        await device.set_gas(gas[1])
        result = await device.get()
        assert gas[0] == result['gas']


async def test_get_firmware():
    """Confirm the firmware version can be read."""
    async with FlowController(ADDRESS) as device:
        result = await device.get_firmware()
        assert 'v' in result or 'GP' in result


@pytest.mark.parametrize('config', [
    {'up': True, 'down': False, 'zero': True, 'power': False},
    {'up': True, 'down': True, 'zero': False, 'power': True},
    {'up': False, 'down': False, 'zero': False, 'power': False},
    ])
async def test_ramp_config(config):
    """Confirm changing the ramping configuration works."""
    async with FlowController(ADDRESS) as device:
        await device.set_ramp_config(config)
        result = await device.get_ramp_config()
        assert config == result

@pytest.mark.parametrize('control_point',
    ['mass flow', 'vol flow', 'abs pressure', 'gauge pressure', 'diff pressure'])
async def test_control_point(control_point):
    """Confirm changing the control point works."""
    async with FlowController(ADDRESS) as device:
        await device._set_control_point(control_point)
        result = await device._get_control_point()
        assert control_point == result
