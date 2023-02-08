from typing import TypeVar, Protocol
import numpy as np

EPS = 1e-9
"Rounding epsilon"

T = TypeVar('T')

class Interpolable(Protocol[T]):
    def interpolate(self: T, end: T, t: float) -> T:
        ...

def assert_npshape(value: np.ndarray, expected_shape: tuple[int], name: str = None, *, dtype=float):
    value = np.asarray(value, dtype)
    if value.shape == expected_shape:
        return value
    if name is None:
        raise ValueError(f'Invalid shape (expected: {expected_shape}; actual: {value.shape})')
    else:
        raise ValueError(f'Invalid shape for {name} (expected: {expected_shape}; actual: {value.shape})')