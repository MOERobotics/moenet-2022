import { TextureLoader, Texture, Sprite, SpriteMaterial, LoadingManager } from "three";
class AprilTagFamily {
	textures?: Array<Texture>;
	constructor(readonly name: string, private readonly prefix: string, readonly size: number) {
	}

	private texturePaths() {
		return Array.from(Array(this.size).keys())
		.map(i => `images/textures/${this.name.replace('h', '_0')}_${i.toString().padStart(5, '0')}.png`);
	}
	
	loadTextures() {
		const loader = new TextureLoader(new LoadingManager(() => console.log(`Loaded sprite`), undefined, url => console.error(`Unable to load sprite from ${url}`)));

		this.textures = this.texturePaths()
			.map(path => {
				console.log(`Loading apriltag texture from ${path}`);
				return loader.load(path);})
	}
	
	async loadTexturesAsync() {
		const loader = new TextureLoader();

		this.textures = await Promise.all(
			this.texturePaths()
			.map(path => loader.loadAsync(path))
		);
	}
}

export const FAMILY_16h5 = new AprilTagFamily('tag16h5', 'tag16_05', 30);

export class AprilTag {
	constructor(readonly family: AprilTagFamily, readonly id: number) {
		if (id > family.size)
			throw new RangeError();
	}
	get texture(): Texture {
		return this.family.textures![this.id - 1];
	}
}

export class AprilTagDetection {
	readonly sprite: Sprite;
	constructor(readonly tag: AprilTag) {
		const texture = this.tag.texture;
		const material = new SpriteMaterial({
			map: texture,
			color: 0xFF_FF_FF,
		});
		this.sprite = new Sprite(material);
	}
}