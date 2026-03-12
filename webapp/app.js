// app.js - Professional LSC VRM Engine
// Developed for High-Fidelity Sign Language Visualization

let scene, camera, renderer, orbitControls, currentVrm;
let isPlaying = false;
let currentFrame = 0;
let motionData = [];
let clock = new THREE.Clock();

// UI Elements (Modern Glassmorphism)
const UI = {
    status: document.getElementById('status'),
    counter: document.getElementById('frame-counter'),
    btnPlay: document.getElementById('btn-play'),
    btnText: document.getElementById('btn-text'),
    playIcon: document.getElementById('play-icon'),
    badge: document.getElementById('sign-badge')
};

// Bone Cache to avoid repeated .getBoneNode lookups (Performance)
const boneCache = new Map();

initThreeJS();
loadVRM("https://cdn.jsdelivr.net/gh/pixiv/three-vrm@dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm");

function initThreeJS() {
    scene = new THREE.Scene();

    // Camera optimized for upper body LSC view
    camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0.0, 1.4, 2.5);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
    document.body.appendChild(renderer.domElement);

    // Studio Lighting
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));

    orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
    orbitControls.target.set(0, 1.3, 0); // Focus on chest/face
    orbitControls.enableDamping = true;
    orbitControls.update();

    window.addEventListener('resize', onWindowResize);
}

function loadVRM(url) {
    UI.status.innerText = "Sincronizando Avatar VRM...";
    const loader = new THREE.GLTFLoader();
    loader.register((parser) => new THREE.VRMLoaderPlugin(parser));

    loader.load(
        url,
        (gltf) => {
            const vrm = gltf.userData.vrm;
            THREE.VRMUtils.removeUnnecessaryVertices(gltf.scene);
            THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);

            scene.add(vrm.scene);
            currentVrm = vrm;

            // Pose Calibration: Face the camera (Flipped to Math.PI)
            vrm.scene.rotation.y = Math.PI;
            vrm.scene.position.y = -0.8;
            window.currentVrm = vrm; // Expose for external tools/debugging

            UI.status.innerText = "Motor LSC Listo. Cargando secuencias...";
            fetchMotionData();
        },
        null,
        (err) => {
            console.error(err);
            UI.status.innerText = "Falla Crítica: Error de Carga VRM.";
        }
    );
}

async function fetchMotionData() {
    try {
        const response = await fetch('./lsc_motion_dummy.json');
        if (!response.ok) throw new Error("Dataset no encontrado.");

        const data = await response.json();
        const rawFrames = data.frames || data;

        // --- EXPERT DATA MAPPING: Absolute Spatial Calibration ---
        motionData = rawFrames.map(frame => {
            const calibrate = (lm) => {
                if (!lm) return null;
                return lm.map(p => ({
                    ...p,
                    visibility: 1.0,
                    // Map centered [-0.5, 0.5] to [0, 1] screen space
                    x: (p.x || 0) + 0.5,
                    y: (p.y || 0) + 0.5,
                    z: (p.z || 0)
                }));
            };
            return {
                pose: calibrate(frame.poseLandmarks || frame.pose),
                pose3d: calibrate(frame.pose3DLandmarks || frame.pose3d),
                leftHand: calibrate(frame.leftHandLandmarks || frame.leftHand),
                rightHand: calibrate(frame.rightHandLandmarks || frame.rightHand),
                face: calibrate(frame.faceLandmarks || frame.face)
            };
        });

        UI.status.innerText = "Secuencia Tensorial Cargada.";
        UI.badge.innerText = data.label || "LSC Capturado";

        UI.btnPlay.addEventListener('click', () => {
            isPlaying = !isPlaying;
            UI.btnText.innerText = isPlaying ? "Pause Engine" : "Play Motion";
            UI.playIcon.innerText = isPlaying ? "⏸" : "▶";
        });

        animate();
    } catch (e) {
        console.error(e);
        UI.status.innerText = "Error: Sin datos de movimiento.";
    }
}

function animateRig(frame) {
    if (!currentVrm || !frame) return;

    // --- KALIDOKIT SOLVE: High Precision ---
    const poseRig = Kalidokit.Pose.solve(frame.pose, frame.pose3d, {
        runtime: "mediapipe", video: { width: 640, height: 480 }
    });
    const rightHandRig = Kalidokit.Hand.solve(frame.rightHand, "Right");
    const leftHandRig = Kalidokit.Hand.solve(frame.leftHand, "Left");
    const faceRig = Kalidokit.Face.solve(frame.face, {
        runtime: "mediapipe", video: { width: 640, height: 480 }
    });

    // 1. Rig Head/Face
    if (faceRig) {
        rigRotation("head", faceRig.head, 1, 0.5);
    }

    // 2. Rig Body (Pose)
    if (poseRig) {
        // Hips Position (Center of gravity)
        if (poseRig.Hips && poseRig.Hips.position) {
            rigPosition("hips", {
                x: -poseRig.Hips.position.x * 2,
                y: poseRig.Hips.position.y + 0.9,
                z: -poseRig.Hips.position.z
            }, 1, 0.1);
        }
        rigRotation("spine", poseRig.Spine, 1, 0.2);

        // Arms (Symmetry Fix)
        rigRotation("rightUpperArm", poseRig.RightUpperArm, 1, 0.3);
        rigRotation("rightLowerArm", poseRig.RightLowerArm, 1, 0.3);
        rigRotation("leftUpperArm", poseRig.LeftUpperArm, 1, 0.3);
        rigRotation("leftLowerArm", poseRig.LeftLowerArm, 1, 0.3);
    }

    // 3. Rig Hands (Detail)
    if (rightHandRig) rigHand(rightHandRig, "right");
    if (leftHandRig) rigHand(leftHandRig, "left");
}

function rigHand(handRig, side) {
    const prefix = side === "right" ? "right" : "left";
    const wrist = side === "right" ? handRig.RightWrist : handRig.LeftWrist;

    rigRotation(`${prefix}Hand`, wrist, 1, 0.4);

    // Explicit Finger Mapping
    ["Thumb", "Index", "Middle", "Ring", "Little"].forEach(finger => {
        ["Proximal", "Intermediate", "Distal"].forEach(joint => {
            const rigProp = `${side.charAt(0).toUpperCase() + side.slice(1)}${finger}${joint}`;
            if (handRig[rigProp]) {
                rigRotation(`${prefix}${finger}${joint}`, handRig[rigProp], 1, 0.5);
            }
        });
    });
}

function rigRotation(boneName, rot, damp = 1, lerp = 0.3) {
    if (!rot) return;
    if (isNaN(rot.x) || isNaN(rot.y) || isNaN(rot.z)) return;

    let bone = boneCache.get(boneName);
    if (!bone) {
        bone = currentVrm.humanoid.getNormalizedBoneNode ?
            currentVrm.humanoid.getNormalizedBoneNode(boneName) :
            currentVrm.humanoid.getBoneNode(boneName);
        if (bone) boneCache.set(boneName, bone);
    }

    if (!bone) return;

    const euler = new THREE.Euler(rot.x * damp, rot.y * damp, rot.z * damp, rot.rotationOrder || 'XYZ');
    const quat = new THREE.Quaternion().setFromEuler(euler);
    bone.quaternion.slerp(quat, lerp);
}

function rigPosition(boneName, pos, damp = 1, lerp = 0.3) {
    if (!pos) return;
    if (isNaN(pos.x) || isNaN(pos.y) || isNaN(pos.z)) return;

    let bone = boneCache.get(boneName);
    if (!bone) {
        bone = currentVrm.humanoid.getNormalizedBoneNode ?
            currentVrm.humanoid.getNormalizedBoneNode(boneName) :
            currentVrm.humanoid.getBoneNode(boneName);
        if (bone) boneCache.set(boneName, bone);
    }

    if (!bone) return;

    const vec = new THREE.Vector3(pos.x * damp, pos.y * damp, pos.z * damp);
    bone.position.lerp(vec, lerp);
}

function animate(time) {
    requestAnimationFrame(animate);

    if (currentVrm) {
        currentVrm.update(clock.getDelta());
    }

    if (isPlaying && (time - lastTime > 40)) { // ~25 FPS for LSC clarity
        if (motionData.length > 0) {
            animateRig(motionData[currentFrame]);
            currentFrame = (currentFrame + 1) % motionData.length;
            UI.counter.innerText = `Frame: ${currentFrame + 1} / ${motionData.length}`;
        }
        lastTime = time;
    }

    orbitControls.update();
    renderer.render(scene, camera);
}

let lastTime = 0;
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}
