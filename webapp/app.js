// app.js
// LSC Avatar VRM Motion Player
// Maps 1629 dimension numpy features to VRM skeleton via Kalidokit

let scene, camera, renderer, orbitControls, currentVrm;
let isPlaying = false;
let currentFrame = 0;
let motionData = []; // Will hold the 60 frames of dictionary-mapped tensors
let clock = new THREE.Clock();

const UI = {
    status: document.getElementById('status'),
    counter: document.getElementById('frame-counter'),
    btnPlay: document.getElementById('btn-play')
};

initThreeJS();
loadVRM("https://cdn.jsdelivr.net/gh/pixiv/three-vrm@dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm");
// Nota: Usar un avatar de muestra estándar en formato raw desde Github CDN para el MVP para saltar la descarga local obligatoria.

function initThreeJS() {
    // 1. Scene
    scene = new THREE.Scene();

    // 2. Camera
    camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 20.0);
    camera.position.set(0.0, 1.4, 3.0);

    // 3. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);

    // 4. Lights
    const light = new THREE.DirectionalLight(0xffffff, 0.8);
    light.position.set(1.0, 1.0, 1.0).normalize();
    scene.add(light);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    // 5. Controls
    orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
    orbitControls.screenSpacePanning = true;
    orbitControls.target.set(0.0, 1.4, 0.0);
    orbitControls.update();

    window.addEventListener('resize', onWindowResize);
}

function loadVRM(url) {
    UI.status.innerText = "Cargando Avatar VRM...";
    const loader = new THREE.GLTFLoader();

    // Install GLTFLoader plugin for VRM 1.0/2.x
    loader.register((parser) => {
        return new THREE.VRMLoaderPlugin(parser);
    });

    loader.load(
        url,
        (gltf) => {
            const vrm = gltf.userData.vrm;
            if (!vrm) {
                console.error("No VRM structure found in loaded glTF.");
                return;
            }

            THREE.VRMUtils.removeUnnecessaryVertices(gltf.scene);
            THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);

            scene.add(vrm.scene);
            currentVrm = vrm;

            // Rotar para que mire hacia adelante (T-Pose ajustada)
            vrm.scene.rotation.y = Math.PI;
            vrm.scene.position.y = -0.1;

            UI.status.innerText = "Avatar Cargado. Esperando Tensores de Movimiento...";

            // Fetch our LSC data
            fetchMotionData();
        },
        (progress) => {
            UI.status.innerText = `Cargando Avatar... ${Math.round((progress.loaded / progress.total) * 100)}%`;
        },
        (error) => {
            console.error(error);
            UI.status.innerText = "Error cargando Avatar.";
        }
    );
}

// ----------------------------------------------------
// Mock Motion Data Fetching & Parsing 
// ----------------------------------------------------
async function fetchMotionData() {
    try {
        UI.status.innerText = "Cargando Secuencia tensorial de LSC...";
        const response = await fetch('./lsc_motion_dummy.json');

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        motionData = await response.json(); // Array of 60 frames
        UI.status.innerText = "Listo para reproducir.";

        UI.btnPlay.addEventListener('click', () => {
            isPlaying = !isPlaying;
            UI.btnPlay.innerText = isPlaying ? "⏸ Pause" : "▶ Play / Loop";
        });

        // Empezar loop 
        animate();

    } catch (e) {
        console.error("No dummy JSON found. Creating procedural mock for demonstration structure.", e);
        // Si no hay archivo JSON generado, generaremos ruido para que el usuario pueda ver
        // que la estructura de Kalidokit existe y el canvas renderiza vivo.
        motionData = setupProceduralMockData();
        UI.status.innerText = "Listo (Modo Mock)";
        UI.btnPlay.addEventListener('click', () => {
            isPlaying = !isPlaying;
            UI.btnPlay.innerText = isPlaying ? "⏸ Pause" : "▶ Play / Loop";
        });
        animate();
    }
}

// ----------------------------------------------------
// Kalidokit Rigging y Render Loop
// ----------------------------------------------------
function animateRig(frameData) {
    if (!currentVrm || !frameData) return;

    // Calculate IK mathematically based on MediaPipe structural array
    const poseRig = Kalidokit.Pose.solve(frameData.poseLandmarks, frameData.pose3DLandmarks, {
        runtime: "mediapipe", video: { width: 640, height: 480 }
    });
    const rightHandRig = Kalidokit.Hand.solve(frameData.rightHandLandmarks, "Right");
    const leftHandRig = Kalidokit.Hand.solve(frameData.leftHandLandmarks, "Left");
    const faceRig = Kalidokit.Face.solve(frameData.faceLandmarks, {
        runtime: "mediapipe", video: { width: 640, height: 480 }
    });

    rigRotation("Head", poseRig.Head.y, poseRig.Head.x, poseRig.Head.z);

    // Apply pose to Hips/Spine
    rigPosition("Hips", {
        x: -poseRig.Hips.position.x,
        y: poseRig.Hips.position.y + 1,
        z: -poseRig.Hips.position.z
    }, 1, 0.07);

    rigRotation("Spine", poseRig.Spine.y, poseRig.Spine.x, poseRig.Spine.z);

    // Apply Arm IK
    rigRotation("RightUpperArm", poseRig.RightUpperArm.y, poseRig.RightUpperArm.x, poseRig.RightUpperArm.z);
    rigRotation("RightLowerArm", poseRig.RightLowerArm.y, poseRig.RightLowerArm.x, poseRig.RightLowerArm.z);
    rigRotation("LeftUpperArm", poseRig.LeftUpperArm.y, poseRig.LeftUpperArm.x, poseRig.LeftUpperArm.z);
    rigRotation("LeftLowerArm", poseRig.LeftLowerArm.y, poseRig.LeftLowerArm.x, poseRig.LeftLowerArm.z);

    // Apply Hand IK
    if (rightHandRig) {
        rigRotation("RightHand", rightHandRig.RightWrist.y, rightHandRig.RightWrist.x, rightHandRig.RightWrist.z);
        rigRotation("RightRingProximal", rightHandRig.RightRingProximal.y, rightHandRig.RightRingProximal.x, rightHandRig.RightRingProximal.z);
        rigRotation("RightRingIntermediate", rightHandRig.RightRingIntermediate.y, rightHandRig.RightRingIntermediate.x, rightHandRig.RightRingIntermediate.z);
        rigRotation("RightRingDistal", rightHandRig.RightRingDistal.y, rightHandRig.RightRingDistal.x, rightHandRig.RightRingDistal.z);
        rigRotation("RightIndexProximal", rightHandRig.RightIndexProximal.y, rightHandRig.RightIndexProximal.x, rightHandRig.RightIndexProximal.z);
        // ... (resto omitido por brevedad para el index de prueba, Kalidokit mapea a cada hueso).
    }

    if (leftHandRig) {
        rigRotation("LeftHand", leftHandRig.LeftWrist.y, leftHandRig.LeftWrist.x, leftHandRig.LeftWrist.z);
    }
}

// Helpers
function rigRotation(name, x = 0, y = 0, z = 0, dampener = 1, lerpAmount = 0.3) {
    if (!currentVrm) return;
    const Part = currentVrm.humanoid.getBoneNode(THREE.VRMSchema.HumanoidBoneName[name]);
    if (!Part) return;

    let euler = new THREE.Euler(x * dampener, y * dampener, z * dampener, 'XYZ');
    let quaternion = new THREE.Quaternion().setFromEuler(euler);
    Part.quaternion.slerp(quaternion, lerpAmount); // Slerp for smooth transition between 60 fps static slices
}

function rigPosition(name, position, dampener = 1, lerpAmount = 0.3) {
    if (!currentVrm) return;
    const Part = currentVrm.humanoid.getBoneNode(THREE.VRMSchema.HumanoidBoneName[name]);
    if (!Part) return;

    let vector = new THREE.Vector3(position.x * dampener, position.y * dampener, position.z * dampener);
    Part.position.lerp(vector, lerpAmount);
}

// Animation Frame Loop
let lastTime = 0;
function animate(time) {
    requestAnimationFrame(animate);

    // We update VRM
    if (currentVrm) {
        currentVrm.update(clock.getDelta());
    }

    // If playing, advance frame array every ~33ms (approx 30fps)
    if (isPlaying && (time - lastTime > 33)) {
        if (motionData.length > 0) {
            animateRig(motionData[currentFrame]);
            currentFrame = (currentFrame + 1) % motionData.length;
            UI.counter.innerText = `Frame: ${currentFrame + 1} / ${motionData.length}`;
        }
        lastTime = time;
    }

    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ----------------------------------------------------
// Fallback Dummy procedural
// ----------------------------------------------------
function setupProceduralMockData() {
    let mockFrames = [];
    for (let i = 0; i < 60; i++) {
        let pos = Math.sin(i * 0.1) * 0.5;
        let mockHead = [{ x: 0, y: 0.1, z: 0 }];
        let mockPose = [];
        for (let j = 0; j < 33; j++) { mockPose.push({ x: pos, y: pos, z: 0, visibility: 1 }); }
        mockFrames.push({
            poseLandmarks: mockPose,
            pose3DLandmarks: mockPose,
            rightHandLandmarks: mockPose.slice(0, 21),
            leftHandLandmarks: mockPose.slice(0, 21),
            faceLandmarks: mockPose.concat(mockPose)
        });
    }
    return mockFrames;
}
