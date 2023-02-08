import {
	Camera,
	EventDispatcher,
	Quaternion,
	Vector3
} from 'three';
import { ReferenceFrame } from './data';

const _changeEvent = { type: 'change' };
const EPS = 0.000001;

interface FlyControlsConfig {
	movementSpeed: number,
	rollSpeed: number,
	dragToLook: boolean,
	autoForward: boolean,
}

interface MoveState {
	up: number,
	down: number,
	left: number,
	right: number,
	forward: number,
	back: number,
	pitchUp: number,
	pitchDown: number,
	yawLeft: number,
	yawRight: number,
	rollLeft: number,
	rollRight: number
}

export class FlyControls extends EventDispatcher {
	private readonly movementSpeed: number;
	private readonly rollSpeed: number;
	private readonly dragToLook: boolean;
	private readonly autoForward: boolean;

	private status = 0;
	private moveState: MoveState;
	private moveVector: Vector3 = new Vector3(0, 0, 0);
	private rotationVector: Vector3 = new Vector3(0, 0, 0);
	private movementSpeedMult = 1;

	public referenceFrame: ReferenceFrame = { type: 'field' };

	private lastPosition = new Vector3();
	private lastQuaternion = new Quaternion();
	constructor(private readonly object: Camera, private readonly domElement: HTMLElement, options: Partial<FlyControlsConfig> = {}) {
		super();

		const {
			movementSpeed = 1000,
			rollSpeed = Math.PI / 10,
			dragToLook = true,
			autoForward = false,
		} = options;

		this.movementSpeed = movementSpeed;
		this.rollSpeed = rollSpeed;
		this.dragToLook = dragToLook;
		this.autoForward = autoForward;

		this.moveState = {
			up: 0,
			down: 0,
			left: 0,
			right: 0,
			forward: 0,
			back: 0,
			pitchUp: 0,
			pitchDown: 0,
			yawLeft: 0,
			yawRight: 0,
			rollLeft: 0,
			rollRight: 0,
		};

		this.domElement.addEventListener('contextmenu', this.contextmenu);
		this.domElement.addEventListener('pointerdown', this.pointerdown);
		this.domElement.addEventListener('pointermove', this.pointermove);
		this.domElement.addEventListener('pointerup', this.pointerup);

		window.addEventListener('keydown', this.keydown);
		window.addEventListener('keyup', this.keyup);

		this.updateMovementVector();
		this.updateRotationVector();
	}

	private readonly keydown = (event: KeyboardEvent) => {
		if (event.altKey)
			return;

		switch (event.code) {
			case 'ShiftLeft':
			case 'ShiftRight': this.movementSpeedMult = .1; break;

			case 'KeyW': this.moveState.forward = 1; break;
			case 'KeyS': this.moveState.back = 1; break;

			case 'KeyA': this.moveState.left = 1; break;
			case 'KeyD': this.moveState.right = 1; break;

			case 'KeyR': this.moveState.up = 1; break;
			case 'KeyF': this.moveState.down = 1; break;

			case 'ArrowUp': this.moveState.pitchUp = 1; break;
			case 'ArrowDown': this.moveState.pitchDown = 1; break;

			case 'ArrowLeft': this.moveState.yawLeft = 1; break;
			case 'ArrowRight': this.moveState.yawRight = 1; break;

			case 'KeyQ': this.moveState.rollLeft = 1; break;
			case 'KeyE': this.moveState.rollRight = 1; break;

			case 'Digit1':
				this.referenceFrame = { type: 'field' };
				break;
			case 'Digit2':
				this.referenceFrame = { type: 'robot' };
				break;
			case 'Digit3':
				this.referenceFrame = { type: 'camera', id: this.referenceFrame.type === 'camera' ? this.referenceFrame.id + 1 : 0 };
				break;
			case 'Digit4':
				this.referenceFrame = { type: 'tag', id: this.referenceFrame.type === 'tag' ? this.referenceFrame.id + 1 : 0 };
				break;
		}

		this.updateMovementVector();
		this.updateRotationVector();
	}

	private readonly keyup = (event: KeyboardEvent) => {
		switch ( event.code ) {
			case 'ShiftLeft':
			case 'ShiftRight': this.movementSpeedMult = 1; break;

			case 'KeyW': this.moveState.forward = 0; break;
			case 'KeyS': this.moveState.back = 0; break;

			case 'KeyA': this.moveState.left = 0; break;
			case 'KeyD': this.moveState.right = 0; break;

			case 'KeyR': this.moveState.up = 0; break;
			case 'KeyF': this.moveState.down = 0; break;

			case 'ArrowUp': this.moveState.pitchUp = 0; break;
			case 'ArrowDown': this.moveState.pitchDown = 0; break;

			case 'ArrowLeft': this.moveState.yawLeft = 0; break;
			case 'ArrowRight': this.moveState.yawRight = 0; break;

			case 'KeyQ': this.moveState.rollLeft = 0; break;
			case 'KeyE': this.moveState.rollRight = 0; break;
		}

		this.updateMovementVector();
		this.updateRotationVector();
	}

	private readonly pointerdown = (event: PointerEvent) => {
		if (this.dragToLook) {
			this.status++;
		} else {
			switch ( event.button ) {
				case 0: this.moveState.forward = 1; break;
				case 2: this.moveState.back = 1; break;
			}
			this.updateMovementVector();
		}
	};

	private readonly pointermove = (event: PointerEvent) => {
		if ( ! this.dragToLook || this.status > 0 ) {
			const container = this.getContainerDimensions();
			const halfWidth = container.size[ 0 ] / 2;
			const halfHeight = container.size[ 1 ] / 2;

			this.moveState.yawLeft = - ( ( event.pageX - container.offset[ 0 ] ) - halfWidth ) / halfWidth;
			this.moveState.pitchDown = ( ( event.pageY - container.offset[ 1 ] ) - halfHeight ) / halfHeight;

			this.updateRotationVector();
		}
	}

	private readonly pointerup = (event: PointerEvent) => {
		if ( this.dragToLook ) {
			this.status --;
			this.moveState.yawLeft = this.moveState.pitchDown = 0;
		} else {
			switch ( event.button ) {
				case 0: this.moveState.forward = 0; break;
				case 2: this.moveState.back = 0; break;
			}

			this.updateMovementVector();
		}

		this.updateRotationVector();
	}

	private readonly contextmenu = (event: Event) => {
		event.preventDefault();
	}

	update(delta: number) {
		const moveMult = delta * this.movementSpeed * this.movementSpeedMult;
		const rotMult = delta * this.rollSpeed;

		this.object.translateX( this.moveVector.x * moveMult );
		this.object.translateY( this.moveVector.y * moveMult );
		this.object.translateZ( this.moveVector.z * moveMult );

		const tmpQuaternion = new Quaternion( this.rotationVector.x * rotMult, this.rotationVector.y * rotMult, this.rotationVector.z * rotMult, 1 ).normalize();
		this.object.quaternion.multiply(tmpQuaternion);

		if (
			this.lastPosition.distanceToSquared( this.object.position ) > EPS ||
			8 * ( 1 - this.lastQuaternion.dot( this.object.quaternion ) ) > EPS
		) {
			this.dispatchEvent( _changeEvent );
			this.lastQuaternion.copy( this.object.quaternion );
			this.lastPosition.copy( this.object.position );
		}
	};

	private updateMovementVector() {
		const forward = ( this.moveState.forward || ( this.autoForward && ! this.moveState.back ) ) ? 1 : 0;

		this.moveVector.x = ( - this.moveState.left + this.moveState.right );
		this.moveVector.y = ( - this.moveState.down + this.moveState.up );
		this.moveVector.z = ( - forward + this.moveState.back );

		// console.log( 'move:', [ this.moveVector.x, this.moveVector.y, this.moveVector.z ] );
	}

	private updateRotationVector() {
		this.rotationVector.x = ( - this.moveState.pitchDown + this.moveState.pitchUp );
		this.rotationVector.y = ( - this.moveState.yawRight + this.moveState.yawLeft );
		this.rotationVector.z = ( - this.moveState.rollRight + this.moveState.rollLeft );

		//console.log( 'rotate:', [ this.rotationVector.x, this.rotationVector.y, this.rotationVector.z ] );

	};

	private getContainerDimensions() {
		if (this.domElement !== (document as any)) {
			return {
				size: [ this.domElement.offsetWidth, this.domElement.offsetHeight ],
				offset: [ this.domElement.offsetLeft, this.domElement.offsetTop ]
			};
		} else {
			return {
				size: [ window.innerWidth, window.innerHeight ],
				offset: [ 0, 0 ]
			};
		}
	};

	dispose() {
		this.domElement.removeEventListener( 'contextmenu', this.contextmenu );
		this.domElement.removeEventListener( 'pointerdown', this.pointerdown );
		this.domElement.removeEventListener( 'pointermove', this.pointermove );
		this.domElement.removeEventListener( 'pointerup', this.pointerup );

		window.removeEventListener( 'keydown', this.keydown );
		window.removeEventListener( 'keyup', this.keyup );
	};

}