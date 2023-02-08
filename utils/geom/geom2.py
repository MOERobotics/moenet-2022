from typing import Optional
import numpy as np
from .util import Interpolable, EPS

class Rotation2D(Interpolable['Rotation2D']):
    @staticmethod
    def identity():
        return Rotation2D(0)
    
    @classmethod
    def from_point(cls, x: float, y: float) -> 'Rotation2D':
        magnitude = np.hypot(x, y)
        if magnitude < EPS:
            return cls.identity()

        sin = y / magnitude
        cos = x / magnitude
        angle = np.arctan2(sin, cos)
        return Rotation2D(angle, sin, cos)
    
    @staticmethod
    def from_raidans(angle_rad: float) -> 'Rotation2D':
        "Create a `Rotation2D` from an angle in radians"
        return Rotation2D(angle_rad)
    
    @classmethod
    def from_degrees(cls, angle_deg: float) -> 'Rotation2D':
        "Create a `Rotation2D` from an angle in degrees"
        angle_rad: float = np.deg2rad(angle_deg)
        return cls.from_raidans(angle_rad)
    
    @classmethod
    def from_rotations(cls, angle_rot: float) -> 'Rotation2D':
        "Create a `Rotation2D` from an angle in rotations"
        angle_rad: float = 2 * np.pi * angle_rot
        return cls.from_raidans(angle_rad)
    
    angle: float
    "Angle (in radians)"
    sin: float
    "Sine of angle"
    cos: float
    "Cosine of angle"

    def __init__(self, angle: float, sin: Optional[float] = None, cos: Optional[float] = None) -> None:
        angle = float(angle)
        self.angle = angle

        if (sin is None) or (cos is None):
            if (sin is not None) or (cos is not None):
                raise ValueError('Invalid overload')
            self.sin = np.sin(angle)
            self.cos = np.cos(angle)
        else:
            magnitude = np.hypot(sin, cos)
            if magnitude > EPS:
                self.sin = sin / magnitude
                self.cos = cos / magnitude
            else:
                self.sin = 0.0
                self.cos = 1.0
    
    @property
    def tan(self):
        "Tangent of this rotation"
        return self.sin / self.cos
    
    def __add__(self, other: 'Rotation2D') -> 'Rotation2D':
        "Combines both rotations"
        if isinstance(other, Rotation2D):
            return self.rotate_by(other)
        return NotImplemented
    
    def __sub__(self, other: 'Rotation2D') -> 'Rotation2D':
        "Subtracts the new rotation from the current rotation"
        if isinstance(other, Rotation2D):
            return self.rotate_by(-other)
        return NotImplemented
    
    def rotate_by(self, other: 'Rotation2D', *, inv: bool = False) -> 'Rotation2D':
        if inv:
            other = -other
        
        return Rotation2D.from_point(
            self.cos * other.cos - self.sin * other.sin,
            self.cos * other.sin + self.sin * other.cos,
        )
    
    def scale(self, scalar: float) -> 'Rotation2D':
        "Scale angle"
        return Rotation2D(self.angle * scalar)
    
    def __mul__(self, scalar: float) -> 'Rotation2D':
        "Scale angle"
        if isinstance(scalar, (float, int)):
            return self.scale(scalar)
        return NotImplemented
    
    def __truediv__(self, scalar: float) -> 'Rotation2D':
        "Scale angle"
        if isinstance(scalar, (float, int)):
            return self.scale(np.reciprocal(float(scalar)))
        return NotImplemented
    
    def __pos__(self):
        "Identity"
        return self
    
    def __neg__(self):
        "Invert the current rotation"
        return Rotation2D(-self.angle)
    
    def __complex__(self):
        "Convert to a complex number"
        return complex(
            self.sin,
            self.cos,
        )
    
    def to_degrees(self) -> float:
        "Get angle in degrees"
        return np.rad2deg(self.angle)
    
    def to_radians(self) -> float:
        "Get angle in radians"
        return self.angle
    
    def to_rotations(self) -> float:
        "Get angle in rotations"
        return self.angle / (2 * np.pi)
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Rotation2D)
            and np.hypot(self.cos - other.cos, self.sin - other.sin) < EPS
        )
    
    def __hash__(self):
        return hash(self.angle)
    
    def __str__(self):
        return f'Rotation2D({self.angle})'
    
    __repr__ = __str__
    
    def interpolate(self, end: 'Rotation2D', t: float) -> 'Rotation2D':
        if not (0 <= t <= 1):
            raise ValueError('t out of bounds')
        elif t == 0:
            return self
        elif t == 1:
            return end
        else:
            delta = (end - self)
            return self + (delta * t)

class Translation2D(Interpolable['Translation2D']):
    @staticmethod
    def identity():
        "Identity translation"
        return Translation2D(0, 0)

    @staticmethod
    def from_polar(r: float, theta: 'Rotation2D'):
        "Create a translation from polar coordinates"
        return Translation2D(
            r * theta.cos,
            r * theta.sin,
        )

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
    
    def distance_to(self, other: 'Translation2D') -> float:
        "Distance between this translation and another one"
        return np.hypot(self.x - other.x, self.y - other.y)

    def norm(self) -> float:
        "Magnitude of this translation (distance from origin)"
        return np.hypot(self.x, self.y)

    def rotate_by(self, rotation: Rotation2D) -> 'Translation2D':
        "Rotate this translation about the origin"
        s = rotation.sin
        c = rotation.cos
        return Translation2D(
            self.x * c - self.y * s,
            self.x * s + self.y * c
        )

    def scale(self, scalar: float) -> 'Translation2D':
        "Scale translation"
        scalar = float(scalar)
        return Translation2D(self.x * scalar, self.y * scalar)

    def __add__(self, other: 'Translation2D') -> 'Translation2D':
        "Combine translations"
        if isinstance(other, Translation2D):
            return Translation2D(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other: 'Translation2D') -> 'Translation2D':
        "Find difference between translations"
        if isinstance(other, Translation2D):
            return Translation2D(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __neg__(self) -> 'Translation2D':
        "Invert this translation"
        return Translation2D(-self.x, -self.y)
    
    def __mul__(self, scalar: float) -> 'Translation2D':
        "Scale translation"
        if isinstance(scalar, (float, int)):
            return self.scale(scalar)
        return NotImplemented
    
    def __truediv__(self, scalar: float) -> 'Translation2D':
        if isinstance(scalar, (float, int)):
            return self.scale(np.reciprocal(float(scalar)))
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __str__(self):
        return f'Translation2D(x={self.x}, y={self.y})'
    
    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Translation2D)
            and np.abs(self.x - other.x) < EPS
            and np.abs(self.y - other.y) < EPS
        )
    
    def interpolate(self, end: 'Translation2D', t: float) -> 'Translation2D':
        if not 0 < t < 1:
            raise ValueError('Out of bounds')
        elif t == 0:
            return self
        elif t == 1:
            return end
        else:
            return Translation2D(
                self.x * (1. - t) + end.x * t,
                self.y * (1. - t) + end.y * t,
            )

class Transform2D:
    @staticmethod
    def identity():
        "Identity transform"
        return Transform2D(
            Translation2D.identity(),
            Rotation2D.identity(),
        )
    
    @staticmethod
    def between(initial: 'Pose2D', last: 'Pose2D'):
        return last - initial

    def __init__(self, translation: Translation2D, rotation: Rotation2D):
        self.translation = translation
        self.rotation = rotation
    
    @property
    def x(self):
        "X component of translation"
        return self.translation.x
    
    @property
    def y(self):
        "Y component of translation"
        return self.translation.y

    def __add__(self, other: 'Transform2D') -> 'Transform2D':
        "Compose two transformations"
        if isinstance(other, Transform2D):
            return Transform2D.between(
                Pose2D.zero(),
                (Pose2D.zero() + self) + other
            )
        return NotImplemented

    def __sub__(self, other: 'Transform2D') -> 'Transform2D':
        if isinstance(other, Transform2D):
            return self + (-other)
        return NotImplemented
    
    def __neg__(self) -> 'Transform2D':
        "Invert the translation"
        inv_rot = -self.rotation
        return Transform2D(
            (-self.translation).rotate_by(inv_rot),
            inv_rot
        )
    
    def __mul__(self, scalar: float) -> 'Transform2D':
        return Transform2D(
            self.translation * scalar,
            self.rotation * scalar,
        )
    
    def __truediv__(self, scalar: float) -> 'Translation2D':
        return self * (1. / scalar)

class Twist2D:
    """
    A change in distance along a 2D arc
    """

    def __init__(self, dx: float, dy: float, dtheta: float) -> None:
        self.dx = float(dx)
        self.dy = float(dy)
        self.dtheta = float(dtheta)
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Twist2D)
            and np.abs(self.dx - other.dx) < EPS
            and np.abs(self.dy - other.dy) < EPS
            and np.abs(self.dtheta - other.dtheta) < EPS
        )
    
    def __str__(self):
        return f'Twist2D(dx={self.dx}, dy={self.dy}, dtheta={self.dtheta})'
    
    __repr__ = __str__

class Pose2D:
    @staticmethod
    def zero():
        return Pose2D(Translation2D.identity(), Rotation2D.identity())
    
    @staticmethod
    def from_transform(transform: Transform2D):
        return Pose2D(
            transform.translation,
            transform.rotation,
        )

    translation: Translation2D
    "Pose translation"
    rotation: Rotation2D
    "Pose orientation"

    def __init__(self, translation: Translation2D, rotation: Rotation2D):
        self.translation = translation
        self.rotation = rotation
    
    @property
    def x(self):
        "X component of position"
        return self.translation.x
    
    @property
    def y(self):
        "Y component of position"
        return self.translation.y

    def relative_to(self, other: 'Pose2D'):
        transform = self - other
        return Pose2D(
            transform.translation,
            transform.rotation,
        )
    
    def exp(self, twist: Twist2D) -> 'Pose2D':
        """
        Obtain a new Pose2d from a (constant curvature) velocity.
        
        See [Controls Engineering in the FIRST Robotics Competition](https://file.tavsys.net/control/controls-engineering-in-frc.pdf)
        section 10.2 "Pose exponential" for a derivation.
        
        The twist is a change in pose in the robot's coordinate frame since the previous pose
        update. When the user runs exp() on the previous known field-relative pose with the argument
        being the twist, the user will receive the new field-relative pose.
        
        "Exp" represents the pose exponential, which is solving a differential equation moving the
        pose forward in time.
        
        ## Parameters
         - `twist` The change in pose in the robot's coordinate frame since the previous pose update.
            For example, if a non-holonomic robot moves forward 0.01 meters and changes angle by 0.5
            degrees since the previous pose update, the twist would be Twist2d(0.01, 0.0,
            Units.degreesToRadians(0.5)).
        ## Returns
        The new pose of the robot.
        """
        dtheta = twist.dtheta

        sinTheta = np.sin(dtheta)
        cosTheta = np.cos(dtheta)

        if np.abs(dtheta) < EPS:
            s = 1 - 1 / 6 * dtheta * dtheta
            c = 0.5 * dtheta
        else:
            s = sinTheta / dtheta
            c = (1 - cosTheta) / dtheta
        
        dx = twist.dx
        dy = twist.dy
        transform = Transform2D(
            Translation2D(dx * s - dy * c, dx * c + dy * s),
            Rotation2D.from_point(cosTheta, sinTheta)
        )
        return self + transform

    def log(self, end: 'Pose2D') -> Twist2D:
        "Returns a Twist2d that maps this pose to the end pose. If c is the output of {@code a.Log(b)}, then {@code a.Exp(c)} would yield b."
        transform = end.relative_to(self)
        dtheta = transform.rotation.to_radians()
        halfDtheta = dtheta / 2

        cosMinusOne = transform.rotation.cos - 1

        if np.abs(cosMinusOne) < EPS:
            halfThetaByTanOfHalfDtheta = 1 - 1 / 12 * dtheta * dtheta
        else:
            halfThetaByTanOfHalfDtheta = -(halfDtheta * transform.rotation.sin) / cosMinusOne

        translationPart = (
            transform.translation
            .rotate_by(Rotation2D.from_point(halfThetaByTanOfHalfDtheta, -halfDtheta))
            * np.hypot(halfThetaByTanOfHalfDtheta, halfDtheta)
        )

        return Twist2D(translationPart.x, translationPart.y, dtheta)
    
    def __add__(self, transform: Transform2D) -> 'Pose2D':
        return Pose2D(
            self.translation + (transform.translation.rotate_by(self.rotation)),
            transform.rotation + self.rotation
        )

    def __sub__(self, other: 'Pose2D') -> Transform2D:
        if not isinstance(other, Pose2D):
            return NotImplemented
        
        return Transform2D(
            (self.translation - other.translation).rotate_by(-other.rotation),
            self.rotation - other.rotation
        )
    
    def __str__(self) -> str:
        return f'Pose2D({self.translation}, {self.rotation})'
    
    def __repr__(self):
        return f'Pose2D(translation={repr(self.translation)}, rotation={repr(self.rotation)})'
    
    def __hash__(self) -> int:
        return hash((self.translation, self.rotation))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Pose2D)
            and self.translation == other.translation
            and self.rotation == other.rotation
        )