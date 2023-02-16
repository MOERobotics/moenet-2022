import { AxesHelper, BoxGeometry, Material, Mesh, MeshBasicMaterial, Quaternion, Scene, Texture, TextureLoader, Vector3 } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { AprilTag, FAMILY_16h5 } from "./apriltag";
import DataSource, { CameraId, ItemId, Pose3D, ReferenceFrame, TagId } from "./data";
const ROBOT_HEIGHT = 1.0;
/** Width of field (m) */
const FIELD_WIDTH = 15.98;
/** Height of field (m) */
const FIELD_HEIGHT = 8.21;
/** Length of a side of an AprilTag (mm) */
const TAG_SIZE = .152;
/** Show an AxesHelper with each item */
const SHOW_AXES = true;
const robotGeometry = new BoxGeometry(ROBOT_HEIGHT, 1.0, 1.0);
const cameraGeometry = new BoxGeometry(.110, .024, .033);
const tagGeometry = new BoxGeometry(.001, TAG_SIZE, TAG_SIZE);
const fieldGeometry = new BoxGeometry(FIELD_WIDTH, FIELD_HEIGHT, .001);

/** Equality check for reference frames */
function matchFrame(u: ReferenceFrame, v: ReferenceFrame) {
	if (u.type !== v.type)
		return false;
	return (u.type == 'robot' || u.type == 'field') || (u.id == (v as typeof u).id);
}

function hashId(item: ItemId): string {
	switch (item.type) {
		case 'robot':
			return item.type;
		case 'camera':
		case 'tag':
			return `${item.type}:${item.id}`;
	}
}

function getMaterials(mesh: Mesh): Material[] {
	return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

function setOpacity(mesh: Mesh, opacity: number) {
	for (const material of getMaterials(mesh)) {
		material.transparent = opacity !== 1;
		material.opacity = opacity;
	}
}

const RETAIN_FRAMES = 100;

interface ItemMeshes {
	main: Mesh;
	axes?: AxesHelper;
	lastSeen: number;
}

export class RFView {
	private _frame: ReferenceFrame = { type: 'camera', id: 0 };
	private frameId = 0;
	private items = new Map<string, ItemMeshes>();
	private readonly robotTexture: Texture;
	private readonly cameraTexture: Texture;
	private readonly fieldTexture: Texture;

	private rfMesh?: Mesh = undefined;

	constructor(private readonly scene: Scene) {
		const loader = new TextureLoader().setPath('images/textures/');
		FAMILY_16h5.loadTextures();
		this.robotTexture = loader.load("crate.gif");
		this.cameraTexture = loader.load("oakd-s2.png");
		this.fieldTexture = loader.load("2023-field.png");

		const modelLoader = new GLTFLoader().setPath('models/');
		modelLoader.load('field-2023.glb', gltf => {
			console.log(gltf.scene);
			this.scene.add(gltf.scene);
		}, e => {
			console.log(e.loaded / e.total);
		});
	}

	private validItems(data: DataSource): Array<ItemId & Pose3D> {
		const items = data.items
			.map(({ poses, ...id }) => {
				const pose = poses.find(({ frame }) => matchFrame(frame, this.frame));
				if (pose === undefined)
					return undefined;
				const { translation, rotation } = pose;
				return { translation, rotation, ...id } as ItemId & Pose3D;
			})
			.filter((item): item is Pose3D & ItemId => item !== undefined) as Array<ItemId & Pose3D>;
		return items;
	}

	private makeItem(item: ItemId) {
		console.log(`Adding ${item.type}`);
		switch (item.type) {
			case 'robot': {
				const material = new MeshBasicMaterial({ map: this.robotTexture });
				material.transparent = true;
				material.opacity = .5;
				return new Mesh(robotGeometry, material);
			}
			case 'tag': {
				const texture = new AprilTag(FAMILY_16h5, item.id).texture;
				const cubeMaterials = [
					new MeshBasicMaterial({map: texture }),
					new MeshBasicMaterial({map: this.robotTexture }),
					new MeshBasicMaterial({map: this.robotTexture }),
					new MeshBasicMaterial({map: this.robotTexture }),
					new MeshBasicMaterial({map: this.robotTexture }),
					new MeshBasicMaterial({map: this.robotTexture }),
				];
				return new Mesh(tagGeometry, cubeMaterials);
			}
			case 'camera': {
				const material = new MeshBasicMaterial({ map: this.cameraTexture });
				return new Mesh(cameraGeometry, material);
			}
		}
	}

	private buildReferenceFrame(rf: ReferenceFrame): Mesh {
		switch (rf.type) {
			case 'robot':
			case 'camera':
			case 'tag': {
				const mesh = this.makeItem(rf);
				setOpacity(mesh, .5);
				mesh.position.set(0,0,0);
				return mesh;
			}
			case 'field': {
				const material = new MeshBasicMaterial({ map: this.fieldTexture });
				const mesh = new Mesh(fieldGeometry, material);
				mesh.position.set(FIELD_WIDTH / 2, FIELD_HEIGHT / 2, 0);
				return mesh;
			}
		}
	}

	get frame() {
		return this._frame;
	}

	setFrame(frame: ReferenceFrame) {
		if (matchFrame(frame, this._frame))
			return false;
		console.log(`Setting reference frame to:`, frame);
		this._frame = frame;

		for (const { main, axes } of this.items.values()) {
			this.scene.remove(main);
			if (axes)
				this.scene.remove(axes);
		}
		this.items.clear();
		if (this.rfMesh !== undefined) {
			this.scene.remove(this.rfMesh);
			this.rfMesh = undefined;
		}

		this.rfMesh = this.buildReferenceFrame(this._frame);
		if (this.rfMesh !== undefined) {
			this.scene.add(this.rfMesh);
		}

		return true;
	}

	update(data: DataSource) {
		const validItems = this.validItems(data);
		this.frameId++;

		if (this._frame.type === 'camera' && this._frame.id !== 0) {
			// Wrap around for camera reference frame
			const maxCameraId = validItems
				.filter((it): it is CameraId & Pose3D => it.type === 'camera')
				.map(cam => cam.id)
				.reduce((cam_id, acc) => cam_id > acc ? cam_id : acc, -Infinity);
			
			if (Number.isFinite(maxCameraId) && this._frame.id > maxCameraId) {
				this.setFrame({type: this._frame.type, id: this._frame.id % maxCameraId });
			}
		} else if (this._frame.type === 'tag' && this._frame.id > FAMILY_16h5.size) {
			// Wrap around for tag reference frame
			this.setFrame({type: this._frame.type, id: (this._frame.id % FAMILY_16h5.size) });
		}

		const missedItems = new Set(this.items.keys());
		for (const item of validItems) {
			const hash = hashId(item);
			missedItems.delete(hash);

			// We need to modify some types of objects
			let { translation, rotation } = item;
			switch (item.type) {
				case 'robot':
					// Offset robot so that the bottom is at the floor
					translation = new Vector3(0, 0, ROBOT_HEIGHT / 2).add(translation);
					break;
			}

			const extant = this.items.get(hash);
			if (extant !== undefined) {
				const { main, axes } = extant;
				if (item.type !== 'robot') {
					for (const material of getMaterials(main)) {
						material.transparent = false;
						material.opacity = 1;
					}
				}
				main.position.copy(translation);
				main.quaternion.copy(rotation);

				if (axes) {
					axes.position.copy(translation);
					axes.quaternion.copy(rotation);
				}
				
				
				// Set as last seen
				extant.lastSeen = this.frameId;
			} else {
				// Create new field object
				const main = this.makeItem(item)!;
				const axes = SHOW_AXES ? new AxesHelper() : undefined;
				this.items.set(hash, { main, axes, lastSeen: this.frameId });

				// Set its position
				main.position.copy(translation);
				main.quaternion.copy(rotation);

				this.scene.add(main);

				if (axes) {
					axes.position.copy(translation);
					axes.quaternion.copy(rotation)
					this.scene.add(axes);
				}
			}
		}

		// Fade items that we didn't see this time
		for (const itemName of missedItems.values()) {
			const item = this.items.get(itemName);
			if (item === undefined)
				continue;
			const { main, axes, lastSeen } = item;
			const framesSinceSeen = this.frameId - lastSeen;
			if (framesSinceSeen > RETAIN_FRAMES) {
				this.scene.remove(main);
				if (axes)
					this.scene.remove(axes);
				this.items.delete(itemName);
				continue;
			}
			for (const material of getMaterials(main)) {
				material.transparent = true;
				material.opacity = 1 - (framesSinceSeen / RETAIN_FRAMES);
			}
		}
	}
}