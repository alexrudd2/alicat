"""Test the BASIS devices/driver respond with correct data."""
from random import uniform
from unittest import mock

import pytest

from alicat.basis import BASISController, BASISMeter
from alicat.mock import BASISClient

ADDRESS = "COM16" # tests requite unit: A, baud: 38400

@pytest.fixture(scope='module', autouse=True)
async def patch_serial_client(request):
    """Replace the serial client with our mock."""
    if request.config.getoption("--hardware"):
        async with BASISMeter(ADDRESS) as device:
            res = await device._write_and_read('A')
            if not res:
                pytest.exit("Device not found. Ensure device is connected " \
                "with unit id A and BAUD rate 38400 on the correct COM port.")
        yield
        async with BASISMeter(ADDRESS) as device:
            await device._write_and_read('AS 0')
    else:
        with mock.patch('alicat.basis.SerialClient', BASISClient):
            yield

@pytest.mark.skip
@pytest.mark.parametrize('cls', [BASISMeter, BASISController])  # Fixme
async def test_is_connected(cls):
    """Confirm that connection status works."""
    async with cls(ADDRESS) as device:
        assert await device.is_connected(ADDRESS)
        assert not await device.is_connected('bad_address')

async def test_tare_flow():
    """Confirm taring the flow works."""
    async with BASISMeter(ADDRESS) as device:
        await device.tare()
        result = await device.get()
        assert result['mass_flow'] == 0.0

async def test_reset_totalizer():
    """Confirm resetting the totalizer works."""
    async with BASISController(ADDRESS) as device:
        await device.reset_totalizer()
        result = await device.get()
        assert result['totalizer'] == 0.0

@pytest.mark.parametrize('gas', ['Air', 'H2'])
async def test_set_standard_gas_name(gas):
    """Confirm that setting standard gases by name works."""
    async with BASISController(ADDRESS) as device:
        await device.set_gas(gas)
        result = await device.get()
        assert gas == result['gas']
        with pytest.raises(ValueError, match='not supported'):
            await device.set_gas('methylacetylene-propadiene propane')

async def test_get_set_pid_terms():
    """Confirm PID terms are updated properly."""
    async with BASISController(ADDRESS) as device:
        p = round(uniform(100, 500))
        i = round(uniform(1000, 5000))
        await device.set_pid(p, i)
        result = await device.get_pid()
        assert {"P": f'{p}', "I": f'{i}'} == result

async def test_totalizer_batch_volume():
    """Confirm setting the totalizer batch volume works."""
    async with BASISController(ADDRESS) as device:
        batch_vol = round(uniform(1, 100))
        await device.set_totalizer_batch(batch_vol)
        result = await device.get_totalizer_batch()
        assert batch_vol == pytest.approx(float(result[0]))

async def test_valve_hold():
    """Confirm holding at a given valve drive percentage works."""
    async with BASISController(ADDRESS) as device:
        valve_drive = round(uniform(1, 100))
        await device.hold(valve_drive)
        result = await device.get()
        assert "HLD" in result['control_point']

        await device.cancel_hold()
        result = await device.get()
        assert "HLD" not in result['control_point']

@pytest.mark.parametrize('gas', [('Air', 0), ('Ar', 1)])
async def test_set_standard_gas_number(gas):
    """Confirm that setting standard gases by number works."""
    async with BASISController(ADDRESS) as device:
        await device.set_gas(gas[1])
        result = await device.get()
        assert gas[0] == result['gas']

async def test_flow_setpoint_roundtrip():
    """Confirm that setting/getting flowrates works."""
    async with BASISController(ADDRESS) as device:
        flow_sp = round(uniform(1, 100), 2)
        await device.set_flow_rate(flowrate=flow_sp)
        result = await device.get()
        assert flow_sp == pytest.approx(result['setpoint'], 0.1)
