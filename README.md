alicat
======

Serial python driver for [Alicat mass flow controllers](http://www.alicat.com/products/mass-flow-meters-and-controllers/mass-flow-controllers/).

![](http://www.alicat.com/wpinstall/wp-content/uploads/2013/12/MCD-266_12001.jpg)

Installation
============

#####TODO Make pypi

This depends on [pyserial](http://pyserial.sourceforge.net/).

Usage
=====

```python
from alicat import FlowController
flow_controller = FlowController(port="/dev/ttyUSB0")
print(flow_controller.get())
```

If the flow controller is communicating on the specified port, this should
return a dictionary of the form:

```python
{
  "flow_setpoint": 0.0,  # Mass flow setpoint
  "gas": "Air",          # Can be any option in `flow_controller.gases`
  "mass_flow": 0.0,      # Mass flow (in units specified at time of purchase)
  "pressure": 25.46,     # Pressure (normally in psia)
  "temperature": 23.62,  # Temperature (normally in C)
  "total_flow": 0.0,     # Optional. If totalizer function purchased, will be included
  "volumetric_flow": 0.0 # Volumetric flow (in units specified at time of purchase)
}
```

You can also set the gas type and flow rate.

```python
flow_controller.set_gas("N2")
flow_controller.set_flow_rate(1.0)
```
