import { BoxGeometry, Material, Mesh, MeshBasicMaterial, Quaternion, Scene, Sprite, SpriteMaterial, Texture, TextureLoader, Vector3 } from "three";
import { AprilTag, FAMILY_16h5 } from "./apriltag";
import DataSource, { CameraId, ItemId, Pose3D, ReferenceFrame, TagId } from "./data";
const ROBOT_HEIGHT = 1.0;
const FIELD_WIDTH = 15.98;
const FIELD_HEIGHT = 8.21;
const robotGeometry = new BoxGeometry(ROBOT_HEIGHT, 1.0, 1.0);
const cameraGeometry = new BoxGeometry(.110, .024, .033);
const tagGeometry = new BoxGeometry(.001, .152, .152);
const fieldGeometry = new BoxGeometry(FIELD_WIDTH, FIELD_HEIGHT, .01);

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

function setOpacity(mesh: Mesh, opacity: number) {
	const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
	for (const material of materials) {
		material.transparent = opacity !== 1;
		material.opacity = opacity;
	}
}

const RETAIN_FRAMES = 100;

export class RFView {
	private _frame: ReferenceFrame = { type: 'camera', id: 0 }
	private frameId = 0;
	private items = new Map<string, [mesh: Mesh | Sprite, lastSeen: number]>();
	private readonly robotTexture: Texture;
	private readonly cameraTexture: Texture;
	private readonly fieldTexture: Texture;

	private rfMesh?: Mesh = undefined;

	constructor(private readonly scene: Scene) {
		const loader = new TextureLoader();
		FAMILY_16h5.loadTextures();
		this.robotTexture = loader.load("images/textures/crate.gif");
		this.cameraTexture = loader.load("images/textures/oakd-s2.png");
		this.fieldTexture = loader.load("images/textures/2023-field.png");
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
				const material = new MeshBasicMaterial({ map: texture });
				return new Mesh(tagGeometry, material);
			}
			case 'camera': {
				const material = new MeshBasicMaterial({ map: this.cameraTexture });
				material.transparent = false;
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

		for (const [item, _] of this.items.values()) {
			this.scene.remove(item);
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
			const maxCameraId = validItems
				.filter((it): it is CameraId & Pose3D => it.type === 'camera')
				.map(cam => cam.id)
				.reduce((cam_id, acc) => cam_id > acc ? cam_id : acc, -Infinity);
			
			if (Number.isFinite(maxCameraId) && this._frame.id > maxCameraId) {
				this.setFrame({type: this._frame.type, id: this._frame.id % maxCameraId });
			}
		} else if (this._frame.type === 'tag' && this._frame.id > FAMILY_16h5.size) {
			this.setFrame({type: this._frame.type, id: ((this._frame.id - 1) % (FAMILY_16h5.size - 1)) + 1 });
		}

		const missedItems = new Set(this.items.keys());
		for (const item of validItems) {
			const hash = hashId(item);
			missedItems.delete(hash);
			const extant = this.items.get(hash);
			let translation = item.translation;
			switch (item.type) {
				case 'robot':
					// Offset robot so that the bottom is at the floor
					translation = new Vector3(0, 0, ROBOT_HEIGHT / 2).add(translation);
					break;
			}

			if (extant !== undefined) {
				const mesh = extant[0];
				const material = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material);
				material.transparent = false;
				material.opacity = 1;
				mesh.position.copy(translation);
				mesh.quaternion.copy(item.rotation);
				
				extant[1] = this.frameId;
			} else {
				const mesh = this.makeItem(item)!;
				this.items.set(hash, [mesh, this.frameId]);
				mesh.position.copy(translation);
				mesh.quaternion.copy(item.rotation);
				this.scene.add(mesh);
			}
		}

		for (const itemName of missedItems.values()) {
			const item = this.items.get(itemName);
			if (item === undefined)
				continue;
			const [mesh, lastSeen] = item;
			const framesSinceSeen = this.frameId - lastSeen;
			if (framesSinceSeen > RETAIN_FRAMES) {
				this.scene.remove(mesh);
				this.items.delete(itemName);
				continue;
			}
			let material = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material);
			material.transparent = true;
			material.opacity = 1 - (framesSinceSeen / RETAIN_FRAMES);
		}
	}
}