from typing import Optional, overload, Union, NamedTuple, Literal, TYPE_CHECKING
import numpy as np
from .geom2 import Rotation2D, Translation2D, Pose2D
from .util import Interpolable, EPS, assert_npshape
from .quaternion import Quaternion

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation as ScipyRotation3D


class AxisAngle(NamedTuple):
    axis: np.ndarray
    angle: float


class Rotation3D(Interpolable['Rotation3D']):
    @staticmethod
    def identity():
        "Identity rotation"
        return Rotation3D(Quaternion.identity())
    
    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float, *, degrees: bool = False) -> 'Rotation3D':
        """
        Constructs a Rotation3d from extrinsic roll, pitch, and yaw.
        
        Extrinsic rotations occur in that order around the axes in the fixed global frame rather
        than the body frame.
        
        Angles are measured counterclockwise with the rotation axis pointing "out of the page". If
        you point your right thumb along the positive axis direction, your fingers curl in the
        direction of positive rotation.
        
        ## Parameters
         - `roll` The counterclockwise rotation angle around the X axis (roll) in radians.
         - `pitch` The counterclockwise rotation angle around the Y axis (pitch) in radians.
         - `yaw` The counterclockwise rotation angle around the Z axis (yaw) in radians.
         - `degrees` If true, the other parameters are in degrees. If false, they're in raidans.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_to_quaternion_conversion
        if degrees:
            roll = np.deg2rad(roll)
            pitch = np.deg2rad(pitch)
            yaw = np.deg2rad(yaw)
        
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        q = Quaternion(
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy)
        
        return Rotation3D(q)
    
    @staticmethod
    def from_axis_angle(axis: 'np.ndarray', angle: float, *, degrees: bool = False) -> 'Rotation3D':
        """
        Constructs a Rotation3d with the given axis-angle representation. The axis doesn't have to be
        normalized.

        ## Parameters
         - `axis` Rotation axis
         - `angle` Rotation around the axis
         - `degrees` If true, angle is in degrees (otherwise it's in radians)
        """
        axis = assert_npshape(axis, (3,), 'axis', dtype=float)
        
        if degrees:
            angle = np.deg2rad(float(angle))
        
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError('Zero axis')

        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Definition
        v = (axis / norm) * np.sin(angle/2)
        return Rotation3D(Quaternion(np.cos(angle/2), v))

    @staticmethod
    def from_scipy(rotation: 'ScipyRotation3D') -> 'Rotation3D':
        pass
    
    @staticmethod
    def from_rotation_matrix(R: 'np.ndarray[float, tuple[Literal[3], Literal[3]]]') -> 'Rotation3D':
        "Constructs a `Rotation3D` from a rotation matrix."
        R = assert_npshape(R, (3,3), 'rotation matrix', dtype=float)

        # Require that the rotation matrix is special orthogonal. This is true if
        # the matrix is orthogonal (RRᵀ = I) and normalized (determinant is 1).
        if np.linalg.norm((R @ R.T) - np.identity(3), ord='fro') > EPS:
            raise ValueError(f"Rotation matrix isn't orthogonal\n\nR =\n{R}")
        if np.abs(np.linalg.det(R) - 1) > EPS:
            raise ValueError(f"Rotation matrix is orthogonal but not special orthogonal\n\nR =\n{R}")

        # Turn rotation matrix into a quaternion
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        diag = np.diagonal(R)
        trace = np.sum(diag)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if diag[0] > diag[1] and diag[0] > diag[2]:
                s = 2 * np.sqrt(1 + diag[0] - diag[1] - diag[2])
                w = (R[2, 1] - R[1, 2]) / s
                x = s / 4
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif diag[1] > diag[2]:
                s = 2 * np.sqrt(1 + diag[1] - diag[0] - diag[2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = s / 4
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2 * np.sqrt(1 + diag[2] - diag[0] - diag[1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = s / 4
        return Rotation3D(Quaternion(w, x, y, z))
    
    @staticmethod
    def from_quaternion(q: Quaternion) -> 'Rotation3D':
        return Rotation3D(q)
    
    @staticmethod
    def between(initial: np.ndarray[Literal[3], float], last: np.ndarray[Literal[3], float]) -> 'Rotation3D':
        """
        Constructs a Rotation3d that rotates the initial vector onto the final vector.
        
        This is useful for turning a 3D vector (final) into an orientation relative to a coordinate
        system vector (initial).
        """
        initial = assert_npshape(initial, (3,), 'initial', dtype=float)
        last = assert_npshape(last, (3,), 'last', dtype=float)
        
        dot = np.dot(initial, last)
        normProduct = np.linalg.norm(initial) * np.linalg.norm(last)
        dotNorm = dot / normProduct

        if dotNorm > (1 - EPS):
            # If the dot product is 1, the two vectors point in the same direction so
            # there's no rotation. The default initialization of m_q will work.
            return Rotation3D.identity()
        elif dotNorm < (-1 + EPS):
            # If the dot product is -1, the two vectors point in opposite directions
            # so a 180 degree rotation is required. Any orthogonal vector can be used
            # for it. Q in the QR decomposition is an orthonormal basis, so it
            # contains orthogonal unit vectors.
            X = initial[:, np.newaxis]
            Q, R = np.linalg.qr(X, mode='complete')

            # w = cos(θ/2) = cos(90°) = 0
            #
            # For x, y, and z, we use the second column of Q because the first is
            # parallel instead of orthogonal. The third column would also work.
            return Rotation3D(Quaternion(0, Q[:, 1]))
        else:
            # initial x last
            axis = np.cross(initial, last)

            # https://stackoverflow.com/a/11741520
            return Rotation3D(
                Quaternion(
                    normProduct + dot,
                    axis,
                ).normalize()
            )

    def __init__(self, /, q: Union[Quaternion, 'Rotation3D']):
        if isinstance(q, Rotation3D):
            self._q = Quaternion(q._q)
        elif isinstance(q, Quaternion):
            self._q = Quaternion(q)
        else:
            raise ValueError('Unknown parameter passed to constructor')
    
    @property
    def x(self) -> float:
        "The counterclockwise rotation angle around the X axis (roll) in radians."
        w = self._q.w
        x = self._q.x
        y = self._q.y
        z = self._q.z

        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    
    @property
    def y(self) -> float:
        "The counterclockwise rotation angle around the Y axis (roll) in radians."
        w = self._q.w
        x = self._q.x
        y = self._q.y
        z = self._q.z

        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        ratio = 2 * (w * y - z * x)
        if np.abs(ratio) >= 1:
            return np.copysign(np.pi / 2, ratio)
        else:
            return np.arcsin(ratio)
    
    @property
    def z(self) -> float:
        "The counterclockwise rotation angle around the Z axis (roll) in radians."
        w = self._q.w
        x = self._q.x
        y = self._q.y
        z = self._q.z
        
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    def rotate_by(self, other: 'Rotation3D', *, inv: bool = False) -> 'Rotation3D':
        if inv:
            return Rotation3D((~other._q) * self._q)
        else:
            return Rotation3D(other._q * self._q)
    
    def scale(self, scalar: float) -> 'Rotation3D':
        # https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
        if self._q.w > 0:
            return Rotation3D.from_axis_angle(
                self._q.vector_part(),
                2 * scalar * np.arccos(self._q.w)
            )
        else:
            return Rotation3D.from_axis_angle(
                -self._q.vector_part(),
                2 * scalar * np.arccos(-self._q.w)
            )

    def __add__(self, other: 'Rotation3D') -> 'Rotation3D':
        "Combines both rotations"
        if isinstance(other, Rotation3D):
            return self.rotate_by(other)
        return NotImplemented
    
    def __sub__(self, other: 'Rotation3D') -> 'Rotation3D':
        "Subtracts the new rotation from the current rotation"
        if isinstance(other, Rotation3D):
            return self.rotate_by(other, inv=True)
        return NotImplemented
    
    def __mul__(self, scalar: Union[float, int]) -> 'Rotation3D':
        "Multiply rotation by scalar"
        if isinstance(scalar, (float, int)):
            return self.scale(float(scalar))
        return NotImplemented
    
    def __truediv__(self, scalar: float) -> 'Rotation3D':
        "Divide rotation by scalar"
        if isinstance(scalar, (float, int)):
            return self.scale(np.reciprocal(float(scalar)))
        return NotImplemented
    
    def __pos__(self):
        "Identity"
        return self
    
    def __neg__(self):
        "Invert the current rotation"
        return Rotation3D(~self._q)
    
    def to_quaternion(self) -> Quaternion:
        "Get rotation as quaternion"
        return self._q
    
    def to_axis_angle(self, *, degrees: bool = False) -> AxisAngle:
        "Get rotation in axis-angle format"
        v = self._q.vector_part()
        norm = np.linalg.norm(v)

        angle = 2 * np.arctan2(norm, self._q.w)
        if degrees:
            angle = np.rad2deg(angle)
        
        if norm == 0:
            axis = np.zeros(3, dtype=float)
        else:
            axis = v / norm
        
        return AxisAngle(axis, angle)
    
    def to_2d(self) -> Rotation2D:
        "Project to 2d plane"
        return Rotation2D(self.z)
    
    def __str__(self):
        return f'Rotation3D({self._q})'
    
    def __repr__(self):
        return f'Rotation3D({repr(self._q)})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rotation3D):
            return False
        return self._q == other._q
    
    def __hash__(self):
        return hash(self._q)
    
    def interpolate(self, end: 'Rotation3D', t: float) -> 'Rotation3D':
        if not (0 <= t <= 1):
            raise ValueError('t out of bounds')
        elif t == 0:
            return self
        elif t == 1:
            return end
        else:
            delta = (end - self)
            return self + (delta * t)

class Translation3D(Interpolable['Translation3D']):
    @staticmethod
    def identity():
        "Identity translation"
        return Translation3D(0, 0, 0)

    @staticmethod
    def from_polar(distance: float, angle: 'Rotation3D'):
        """
        Constructs a Translation3d with the provided distance and angle. This is essentially converting
        from polar coordinates to Cartesian coordinates.
        """
        return Translation3D(distance, 0, 0).rotate_by(angle)

    @overload
    def __init__(self, v: np.ndarray): ...
    @overload
    def __init__(self, x: float, y: float, z: float): ...
    def __init__(self, x: Union[float, np.ndarray], y: Optional[float] = None, z: Optional[float] = None):
        if (y is None) or (z is None):
            if (y is not None) or (z is not None):
                raise ValueError('Invalid overload')
            self._v = assert_npshape(x, (3,), dtype=float)
        else:
            self._v = assert_npshape([x,y,z], (3,), dtype=float)
    
    def distance_to(self, other: 'Translation3D') -> float:
        "Calculates the distance between two translations in 3D space."
        if not isinstance(other, Translation3D):
            raise ValueError()
        
        return np.linalg.norm(self._v - other._v)

    def norm(self) -> float:
        "Compute the distance to the origin"
        return np.linalg.norm(self._v)
    
    @property
    def x(self) -> float:
        "X component of translation"
        return self._v[0]
    
    @x.setter
    def x(self, x: float):
        self._v[0] = x
    
    @property
    def y(self) -> float:
        "Y component of translation"
        return self._v[1]

    @y.setter
    def y(self, y: float):
        self._v[1] = y
    
    @property
    def z(self) -> float:
        "Z component of translation"
        return self._v[2]
    
    @z.setter
    def z(self, z: float):
        self._v[2] = z

    def rotate_by(self, rotation: 'Rotation3D') -> 'Translation3D':
        "Apply a rotation in 3d space"
        p = Quaternion(0, self.x, self.y, self.z)
        q = rotation.to_quaternion()
        qPrime = (q * p) * ~q
        return Translation3D(qPrime.x, qPrime.y, qPrime.z)

    def __add__(self, other: 'Translation3D') -> 'Translation3D':
        "Combine two translations"
        if isinstance(other, Translation3D):
            return Translation3D(self._v + other._v)
        return NotImplemented

    def __sub__(self, other: 'Translation3D') -> 'Translation3D':
        "Difference between the two translations"
        if isinstance(other, Translation3D):
            return Translation3D(self._v - other._v)
        return NotImplemented
    
    def __neg__(self) -> 'Translation3D':
        "Inverse of the translation (equal to negating all the components)"
        return Translation3D(-self._v)
    
    def __mul__(self, scalar: float) -> 'Translation3D':
        "Scale the translation"
        if isinstance(scalar, (float, int)):
            return Translation3D(self._v * float(scalar))
        return NotImplemented
    
    def __truediv__(self, scalar: float) -> 'Translation3D':
        "Scale the translation"
        if isinstance(scalar, (float, int)):
            return Translation3D(self._v / float(scalar))
        return NotImplemented
    
    def as_2d(self) -> Translation2D:
        return Translation2D(
            self._v[0],
            self._v[1]
        )

    def as_vec(self) -> np.ndarray:
        return self._v
    
    def __hash__(self) -> int:
        return hash(self._v)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Translation3D)
            and self.distance_to(other) < EPS
        )
    
    def __str__(self):
        return f'Translation3D({self.x}, {self.y}, {self.z})'
    
    __repr__ = __str__
    
    def interpolate(self, end: 'Translation3D', t: float) -> 'Translation3D':
        if not 0 < t < 1:
            raise ValueError('Out of bounds')
        elif t == 0:
            return self
        elif t == 1:
            return end
        else:
            delta = end - self
            return self + (delta * t)

class Transform3D:
    "Represents a translation between `Pose3D`s"
    @staticmethod
    def identity():
        "Identity transform"
        return Transform3D(
            Translation3D.identity(),
            Rotation3D.identity(),
        )
    
    @staticmethod
    def between(initial: 'Pose3D', last: 'Pose3D') -> 'Transform3D':
        "Compute transform that maps from the initial pose to the last pose"
        return Transform3D(
            (last.translation - initial.translation).rotate_by(-initial.rotation),
            last.rotation - initial.rotation
        )

    def __init__(self, translation: Translation3D, rotation: Rotation3D):
        self.translation = translation
        self.rotation = rotation
    
    @property
    def x(self):
        return self.translation.x
    
    @property
    def y(self):
        return self.translation.y
    
    @property
    def z(self):
        return self.translation.z

    def __add__(self, other: 'Transform3D') -> 'Transform3D':
        "Compose two transformations"
        if isinstance(other, Transform3D):
            # return Transform3D(
            #     self.translation + other.translation.rotate_by(self.rotation),
            #     self.rotation + other.rotation,
            # )
            return Transform3D.between(Pose3D.zero(), Pose3D.zero().transform_by(self).transform_by(other))
        return NotImplemented

    def __sub__(self, other: 'Transform3D') -> 'Transform3D':
        if isinstance(other, Transform3D):
            # return self + (-other)
            pass
        return NotImplemented
    
    def __neg__(self) -> 'Transform3D':
        "Invert the translation"
        inv_rot = -self.rotation
        return Transform3D(
            (-self.translation).rotate_by(inv_rot),
            inv_rot
        )
    
    def __mul__(self, scalar: Union[float, int]) -> 'Transform3D':
        "Scale transform"
        if isinstance(scalar, (float, int)):
            return Transform3D(
                self.translation * scalar,
                self.rotation * scalar,
            )
        return NotImplemented
    
    def __truediv__(self, scalar: float) -> 'Translation3D':
        "Scale transform"
        return self * np.reciprocal(float(scalar))
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Transform3D)
            and self.translation == other.translation
            and self.rotation == other.rotation
        )
    
    def __str__(self):
        return f'Transform3D({self.translation}, {self.rotation})'

class Twist3D:
    "A change in distance along a 3D arc"

    d_translation: np.ndarray[Literal[3], float]
    "Derivative of translation"
    d_rotation: np.ndarray[Literal[3], float]
    "Derivative of rotation"

    def __init__(self, d_translation: np.ndarray[Literal[3], float], d_rotation: np.ndarray[Literal[3], float]) -> None:
        self.d_translation = assert_npshape(np.asarray(d_translation, dtype=float), (3,), "d_translation")
        self.d_rotation    = assert_npshape(np.asarray(   d_rotation, dtype=float), (3,),    "d_rotation")
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Twist3D)
            and np.all(np.abs(self.d_translation - other.d_translation) < EPS)
            and np.all(np.abs(self.d_rotation - other.d_rotation) < EPS)
        )

    def __mul__(self, scale: Union[float, int]) -> 'Twist3D':
        if isinstance(scale, (float, int)):
            return Twist3D(
                self.d_translation * scale,
                self.d_rotation * scale,
            )
        return NotImplemented
    
    def __str__(self):
        return f'Twist3D(d_translation={self.d_translation}, d_rotation={self.d_rotation})'
    __repr__ = __str__

def rotationVectorToMatrix(rotation: np.ndarray):
    # Given a rotation vector <a, b, c>,
    #         [ 0 -c  b]
    # Omega = [ c  0 -a]
    #         [-b  a  0]
    rotation = assert_npshape(rotation, (3,), 'rotation vector', dtype=float)
    return np.array([
        [           0, -rotation[2],  rotation[1]],
        [ rotation[2],            0, -rotation[0]],
        [-rotation[1],  rotation[0],            0],
    ])

class Pose3D(Interpolable['Pose3D']):
    @staticmethod
    def zero():
        return Pose3D(Translation3D.identity(), Rotation3D.identity())
    
    @staticmethod
    def from_transform(transform: Transform3D):
        return Pose3D(
            transform.translation,
            transform.rotation,
        )

    def __init__(self, translation: Translation3D, rotation: Rotation3D):
        self.translation = translation
        self.rotation = rotation
    
    @property
    def x(self):
        return self.translation.x
    
    @property
    def y(self):
        return self.translation.y
    
    @property
    def z(self):
        return self.translation.z

    def relative_to(self, other: 'Pose3D'):
        transform = Transform3D.between(other, self)
        return Pose3D.from_transform(transform)
    
    def exp(self, twist: Twist3D) -> 'Pose3D':
        # Implementation from Section 3.2 of https://ethaneade.org/lie.pdf
        u = twist.d_translation
        rvec = twist.d_rotation
        omega = rotationVectorToMatrix(rvec)
        omegaSq = omega ** 2
        theta = np.linalg.norm(rvec)
        thetaSq = theta * theta

        if theta < EPS:
            # Taylor Expansions around θ = 0
            # A = 1/1! - θ²/3! + θ⁴/5!
            # B = 1/2! - θ²/4! + θ⁴/6!
            # C = 1/3! - θ²/5! + θ⁴/7!
            A = 1 - thetaSq / 6 + thetaSq * thetaSq / 120
            B = 1/2 - thetaSq / 24 + thetaSq * thetaSq / 720
            C = 1/6 - thetaSq / 120 + thetaSq * thetaSq / 5040
        else:
            # A = sin(θ)/θ
            # B = (1 - cos(θ)) / θ²
            # C = (1 - A) / θ²
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / thetaSq
            C = (1 - A) / thetaSq

        R = np.identity(3) + (omega * A) + (omegaSq * B)
        V = np.identity(3) + (omega * B) + (omegaSq * C)
        translation_component = V @ u
        transform = Transform3D(
            Translation3D(translation_component),
            Rotation3D.from_rotation_matrix(R)
        )

        return self + transform

    def log(self, end: 'Pose3D') -> Twist3D:
        "Returns a Twist2d that maps this pose to the end pose. If c is the output of {@code a.Log(b)}, then {@code a.Exp(c)} would yield b."
        # Implementation from Section 3.2 of https://ethaneade.org/lie.pdf
        transform = end.relative_to(self)

        rvec = transform.rotation.to_quaternion().to_rotation_vector()

        omega = rotationVectorToMatrix(rvec)
        _axis, theta = transform.rotation.to_axis_angle()
        thetaSq = theta * theta

        if theta < EPS:
            # Taylor Expansions around θ = 0
            # A = 1/1! - θ²/3! + θ⁴/5!
            # B = 1/2! - θ²/4! + θ⁴/6!
            # C = 1/6 * (1/2 + θ²/5! + θ⁴/7!)
            C = 1 / 6 - thetaSq / 120 + thetaSq * thetaSq / 5040
        else:
            # A = sin(θ)/θ
            # B = (1 - cos(θ)) / θ²
            # C = (1 - A/(2*B)) / θ²
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / thetaSq
            C = (1 - A / (2 * B)) / thetaSq

        V_inv = np.identity(3) - (omega / 2) + ((omega ** 2) * C)

        twist_translation = V_inv @ transform.translation.as_vec()

        return Twist3D(twist_translation, rvec)

    def transform_by(self, other: Transform3D) -> 'Pose3D':
        return Pose3D(
            self.translation + (other.translation.rotate_by(self.rotation)),
            self.rotation + other.rotation
        )
    
    def as_2d(self) -> Pose2D:
        return Pose2D(
            self.translation.as_2d(),
            self.rotation.to_2d(),
        )
    
    def __add__(self, other: Transform3D) -> 'Pose3D':
        if isinstance(other, Transform3D):
            return self.transform_by(other)
        return NotImplemented

    def __sub__(self, other: 'Pose3D') -> Transform3D:
        if isinstance(other, Pose3D):
            return self.relative_to(other)
        return NotImplemented
    
    def __str__(self) -> str:
        return f'Pose3D({self.translation}, {self.rotation})'
    
    def __repr__(self) -> str:
        return f'Pose3D(translation={repr(self.translation)}, rotation={repr(self.rotation)})'
    
    def __hash__(self) -> int:
        return hash((self.translation, self.rotation))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Pose3D)
            and self.translation == other.translation
            and self.rotation == other.rotation
        )
    
    def interpolate(self, end: 'Pose3D', t: float) -> 'Pose3D':
        if not 0 < t < 1:
            raise ValueError('Out of bounds')
        elif t == 0:
            return self
        elif t == 1:
            return end
        else:
            twist = self.log(end)
            scaledTwist = twist * t
            return self.exp(scaledTwist)