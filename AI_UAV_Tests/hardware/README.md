# AI_UAV_Tests Hardware

This folder contains hardware-oriented helpers and motor-mapping utilities.

## Modules

- `motor_mapping.py`
  - Motor mixing and map-construction helpers used by hardware-facing tests.

## Common import

```python
from AI_UAV_Tests.hardware.motor_mapping import make_motor_map_layer
```

## Typical usages

This folder is primarily imported by:

- `Realworld_Deployment/crazyflie_mixer_test.py`
- `Realworld_Deployment/crazyflie_1216_test.py`

Example:

```powershell
python .\Realworld_Deployment\crazyflie_mixer_test.py
```

## Expected outputs

Depending on the caller, expect:

- mixer sanity-check prints
- motor-map diagnostics
- hardware-facing trajectory/controller checks

## Notes

- This folder mostly contains support code rather than standalone experiment reports.
