from .gyro_adam import GYROAdam
from .gyro_sgd import GYROSGD

# Legacy aliases for backward compatibility
CliffordAdam = GYROAdam
CliffordSGD = GYROSGD

__all__ = ['GYROAdam', 'GYROSGD', 'CliffordAdam', 'CliffordSGD']
