import { Quaternion, Vector3 } from "three";

type PoseData = {
	type: string,
	id?: number,
	translation: [number, number, number],
	rotation: [number, number, number, number]
};
type DataUpdate = Array<{
	type: string,
	id?: number,
	poses: Array<PoseData>,
}>

export type TagId = { type: 'tag'; id: number };
export type CameraId = { type: 'camera'; id: number };
export type RobotId = { type: 'robot' };
export type FieldId = { type: 'field' };
export type ReferenceFrame = TagId | CameraId | RobotId | FieldId;
export type ItemId = TagId | CameraId | RobotId;

export interface Pose3D {
	translation: Vector3,
	rotation: Quaternion,
}

export type GameItem = ItemId & {
	poses: Array<Pose3D & {
		frame: ReferenceFrame,
	}>;
}

export default class DataSource {
	private src?: EventSource;
	private _connected: boolean = false;
	private _items: GameItem[] = [
		{
			type: 'robot',
			poses: [
				{
					frame: { type: 'field' },
					translation: new Vector3(0,0,0),
					rotation: new Quaternion(),
				},
				{
					frame: { type: 'robot' },
					translation: new Vector3(0,0,0),
					rotation: new Quaternion(),
				},
			]
		},
		{
			type: 'tag',
			id: 1,
			poses: [
				{
					frame: { type: 'field' },
					translation: new Vector3(15.513558, 1.071626, 0.462788),
					rotation: new Quaternion(0, 0, 1, 0),
				},
				{
					frame: { type: 'camera', id: 0 },
					translation: new Vector3(0, 0, 2),
					rotation: new Quaternion(),
				}
			]
		},
		{
			type: 'tag',
			id: 2,
			poses: [{
				frame: { type: 'field' },
				translation: new Vector3(15.513558, 2.748026, 0.462788),
				rotation: new Quaternion(0, 0, 1, 0),
			}]
		},
		{
			type: 'tag',
			id: 3,
			poses: [{
				frame: { type: 'field' },
				translation: new Vector3(15.513558, 4.424426, 0.462788),
				rotation: new Quaternion(0, 0, 1, 0),
			}]
		},
		{
			type: 'tag',
			id: 4,
			poses: [{
				frame: { type: 'field' },
				translation: new Vector3(16.178784, 6.749796, 0.695452),
				rotation: new Quaternion(0, 0, 1, 0),
			}]
		},
		{
			type: 'tag',
			id: 5,
			poses: [{
				frame: { type: 'field' },
				translation: new Vector3(0.36195, 6.749796, 0.695452),
				rotation: new Quaternion(0, 0, 0, 1),
			}]
		},
		{
			type: 'tag',
			id: 6,
			poses: [{
				frame: { type: 'field' },
				translation: new Vector3(1.02743, 4.424426, 0.462788),
				rotation: new Quaternion(0, 0, 0, 1),
			}]
		},
		{
			type: 'tag',
			id: 7,
			poses: [{
				frame: { type: 'field' },
				translation: new Vector3(1.02743, 2.748026, 0.462788),
				rotation: new Quaternion(0, 0, 0, 1),
			}]
		},
		{
			type: 'tag',
			id: 8,
			poses: [{
				frame: { type: 'field' },
				translation: new Vector3(1.02743, 1.071626, 0.462788),
				rotation: new Quaternion(0, 0, 0, 1),
			}]
		},
	];
	constructor() {
		this.src = new EventSource('/api/data');
		this.reconnect();
	}

	get connected() {
		return this._connected;
	}

	private reconnect() {
		if (this.src !== undefined) {
			this.src.removeEventListener('open', this.handleOpen);
			this.src.removeEventListener('message', this.handleData);
			this.src.removeEventListener('pose_update', this.handleData);
			this.src.removeEventListener('error', this.handleError);
			this.src.close();
		}
		this.src = new EventSource('/api/data');
		this.src.addEventListener('open', this.handleOpen, { once: true });
		this.src.addEventListener('error', this.handleError);
		this.src.addEventListener('message', this.handleData);
		this.src.addEventListener('pose_update', this.handleData);
	}

	private readonly handleOpen = () => {
		this._connected = true;
		console.log('connected');
	};

	private readonly handleError = () => {
		this._connected = false;
		this.reconnect();
	};

	private readonly handleData = (e: MessageEvent) => {
		const data: DataUpdate = JSON.parse(e.data);
		this._items = data.map(({id, type, poses}) => ({
			type,
			id,
			poses: poses.map(({ id, type, translation, rotation }) => ({
				frame: { type, id } as ReferenceFrame,
				translation: new Vector3(translation[0], translation[1], translation[2]),
				rotation: new Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]),
			}))
		}) as GameItem);
	}

	get items(): Array<GameItem> {
		return this._items;
	}
}