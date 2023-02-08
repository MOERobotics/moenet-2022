import { AxesHelper, Clock, PerspectiveCamera, Scene, WebGLRenderer } from "three";
import { FlyControls } from "./controls";
import DataSource from './data';
import { RFView } from "./view";

export class App {
    private clock: Clock;
    private camera: PerspectiveCamera;
    private scene: Scene;
    // private mesh: Mesh;
    private renderer: WebGLRenderer;
    private data = new DataSource();
    private controls: FlyControls;
    private view: RFView;
    private textOverlay: HTMLElement;

    /**
     * Based off the three.js docs: https://threejs.org/examples/?q=cube#webgl_geometry_cube
     */
    constructor() {
        this.clock = new Clock();
        this.camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, .01, 50);

        this.textOverlay = document.createElement('div');
        this.textOverlay.style.position = 'absolute';
        this.textOverlay.style.bottom = '0';
        this.textOverlay.innerText = 'Field';
        document.body.appendChild(this.textOverlay);

        this.scene = new Scene();
        this.scene.add(new AxesHelper(5));

        this.view = new RFView(this.scene);
        this.view.setFrame({ type: 'field' });

        this.renderer = new WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setClearColor(0x000000, 0); // the default
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);

        document.body.appendChild(this.renderer.domElement);

        this.controls = new FlyControls(this.camera, this.renderer.domElement, {
            movementSpeed: 10,
            dragToLook: true
        });

        this.resetCameraAfterRFSwitch();

        window.addEventListener("resize", this.onWindowResize, false);

        this.animate();
    }

    private readonly onWindowResize = () => {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    private resetCameraAfterRFSwitch() {
        const rf = this.view.frame;
        switch (rf.type) {
            case 'field':
                this.camera.position.set(8, -1, 4);
                this.camera.rotation.set(Math.PI / 2, 0, 0);
                this.controls.movementSpeed = 10;
                break;
            case 'robot':
                this.camera.position.set(0, -1.5, 1);
                this.camera.rotation.set(Math.PI / 2, 0, 0);
                this.controls.movementSpeed = 10;
                break;
            case 'camera':
                this.camera.position.set(0, 0, -1);
                this.camera.rotation.set(0,Math.PI,0);
                this.controls.movementSpeed = 1;
                break;
            case 'tag':
                this.camera.position.set(0, -.5, 0);
                this.camera.rotation.set(Math.PI / 2, 0, 0);
                this.controls.movementSpeed = 1;
                break;
        }
    }

    private animate() {
        requestAnimationFrame(() => this.animate());

        if (this.view.setFrame(this.controls.referenceFrame)) {
            this.resetCameraAfterRFSwitch();
        }
        const rf = this.view.frame;
        this.textOverlay.innerText = `${rf.type} ${(rf as any).id ?? ''}`;

        this.render();
    }

    private render() {
        this.view.update(this.data);
        this.controls.update(this.clock.getDelta());
        this.renderer.render(this.scene, this.camera);
    }
}

new App();