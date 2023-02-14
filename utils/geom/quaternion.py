from typing import overload, Union, Literal
import numpy as np
from .util import EPS

class Quaternion:
	@staticmethod
	def zero():
		"Constructs a quaternion with a default angle of 0 degrees."
		return Quaternion(0,0,0,0)
	
	@staticmethod
	def identity():
		"Constructs a quaternion with a default angle of 0 degrees."
		return Quaternion(1.,0,0,0)
	
	_components: np.ndarray[Literal[4], float]

	@overload
	def __init__(self, /, q: 'Quaternion'): ...
	@overload
	def __init__(self, /, components: 'np.ndarray'): ...
	@overload
	def __init__(self, /, w: float, i: float, j: float, k: float): ...
	@overload
	def __init__(self, /, w: float, v: 'np.NDArray'): ...
	def __init__(self, *args):
		"Construct quaternion from parts"
		if len(args) == 1:
			if isinstance(args[0], Quaternion):
				components = np.copy(args[0]._components)
			else:
				components = np.asarray(args[0], dtype=float)
			if components.shape != (4,):
				raise ValueError(f'Expected array of shape (4)')
			self._components = components
		elif len(args) == 2:
			w, v = args
			w = float(w)
			v = np.asarray(v, dtype=float)
			if v.shape != (3,):
				raise ValueError(f'Expected array of shape (3) for imaginary part')
			self._components = np.concatenate([[w], v], dtype=float)
		elif len(args) == 4:
			w, i, j, k = args
			self._components = np.array([w, i, j, k], dtype=float)
		else:
			raise ValueError('Unknown constructor overload')

	@property
	def w(self) -> float:
		"`w` component of quaternion"
		return self._components[0]
	
	@w.setter
	def w(self, w: float):
		self._components[0] = w
	
	@property
	def x(self) -> float:
		"`x` component of quaternion"
		return self._components[1]
	
	@x.setter
	def x(self, x: float):
		self._components[1] = x
	
	@property
	def y(self) -> float:
		"`y` component of quaternion"
		return self._components[2]
	
	@y.setter
	def y(self, y: float):
		self._components[2] = y
	
	@property
	def z(self) -> float:
		"`z` component of quaternion"
		return self._components[3]
	
	@z.setter
	def z(self, z: float):
		self._components[3] = z
	
	def __real__(self) -> float:
		"Get the real component (`w`)"
		return self.w

	def vector_part(self):
		"Get the imaginary vector part (`[i, j, k]`)"
		return self._components[1:]
	
	def is_unit(self, eps: float = EPS) -> bool:
		"Check if this is a unit quaternion"
		return np.abs(self.norm() - 1) < eps
	
	def as_positive_polar(self) -> 'Quaternion':
		if self.w < 0:
			unitQ = self.normalize()
			# The quaternion of rotation (normalized quaternion) q and -q
            # are equivalent (i.e. represent the same rotation).
			return -unitQ
		else:
			return self
	
	def dot(self, other: 'Quaternion') -> float:
		"Dot product between two quaternions"
		return np.dot(self._components, other._components)

	def scale(self, other: Union[float, int]) -> 'Quaternion':
		return Quaternion(self._components * float(other))
	
	def norm(self) -> float:
		"Quaternion norm"
		return np.linalg.norm(self._components)

	def normalize(self) -> 'Quaternion':
		"Normalizes the quaternion."
		norm = self.norm()
		if (norm == 0.0):
			return Quaternion.identity()
		else:
			return self / norm
	
	def conj(self) -> 'Quaternion':
		return Quaternion(self._components * np.array([1., -1., -1., -1.]))
	
	def to_rotation_vector(self, eps: float = EPS):
		"Returns the rotation vector representation of this quaternion."
		# See equation (31) in "Integrating Generic Sensor Fusion Algorithms with
		# Sound State Representation through Encapsulation of Manifolds"
		#
		# https://arxiv.org/pdf/1107.1119.pdf
		w = self.w
		v = self._components[1:]
		norm = np.linalg.norm(v)

		if norm < eps:
			scale = (2 / w) - (2/3) * (norm ** 2) / (w ** 3)
		else:
			if w < 0:
				angle = np.arctan2(-norm, -w)
			else:
				angle = np.arctan2(norm, w)
			scale = 2 * angle / norm
		return v * scale
	
	def __mul__(self, other: Union['Quaternion', float, int]) -> 'Quaternion':
		"Multiply with another quaternion."
		if isinstance(other, (float, int)):
			return self.scale(other)
		elif isinstance(other, Quaternion):
			# https://en.wikipedia.org/wiki/Quaternion#Scalar_and_vector_parts
			# v₁ x v₂
			slf_w = self.w
			slf_v = self.vector_part()
			oth_w = other.w
			oth_v = other.vector_part()
			cross = np.cross(slf_v, oth_v)
			# v = r₁v₂ + r₂v₁ + v₁ x v₂
			v = (oth_v * slf_w) + (slf_v * oth_w) + cross

			return Quaternion(
				slf_w * oth_w - np.dot(slf_v, oth_v),
				v
			)
		else:
			return NotImplemented
	
	def __truediv__(self, other: Union[float, int]) -> 'Quaternion':
		if not isinstance(other, (float, int)):
			return NotImplemented
		return self.scale(np.reciprocal(float(other)))

	def __matmul__(self, other: 'Quaternion') -> float:
		"Dot product between two quaternions"
		if not isinstance(other, Quaternion):
			return NotImplemented

		return self.dot(other)

	def __eq__(self, other: object) -> bool:
		"Checks equality between this Quaternion and another object."
		if not isinstance(other, Quaternion):
			return False
		
		return np.abs(np.dot(self._components, other._components)) > (1.0 - EPS)

	def __hash__(self) -> int:
		return hash(self._components)
	
	def __neg__(self) -> 'Quaternion':
		return Quaternion(self._components * -1)

	def __invert__(self) -> 'Quaternion':
		"Conjugated quaternion"
		return self.conj()
	
	def __str__(self):
		return f'Quaternion(w={self.w}, i={self.x}, j={self.y}, k={self.z})'
	__repr__ = __str__
