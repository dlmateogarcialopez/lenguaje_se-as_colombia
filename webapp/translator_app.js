// translator_app.js — LSC Translator Frontend Engine v5.0
// ENHANCED HUMANOID MANNEQUIN — Direct Position, Zero IK
//
// Architecture:
//   A realistic-looking 3D humanoid mannequin built from Three.js primitives.
//   Each joint is positioned DIRECTLY from MediaPipe landmark coordinates.
//   No IK, no rotation calculation, no translation errors.
//   Features: organic capsule limbs, filled torso, head with eyes, skin materials.

// ─── GLOBALS ─────────────────────────────────────────────────────────────────
let scene, camera, renderer, orbitControls, clock;
let mannequinGroup; // Parent group for all mannequin meshes for easy clearing
let isPlaying = false;
let currentFrame = 0;
let motionData = [];
let lastTime = 0;
const skCanvas = document.getElementById('skeleton-canvas');
const ctx = skCanvas.getContext('2d');
const overlayCanvas = document.getElementById('video-overlay');

const COLORS = {
    body: '#00EDDA', rhand: '#ff6b6b', lhand: '#6bcbff',
    face: '#c0f566', spine: '#ffd700', joints: '#fff',
};

const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [11, 12], [11, 23], [12, 24], [23, 24],
    [11, 13], [13, 15], [12, 14], [14, 16],
    [15, 17], [15, 19], [15, 21], [17, 19],
    [16, 18], [16, 20], [16, 22], [18, 20],
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17],
];
const API_BASE = "";
const UI = {
    status: document.getElementById('status'),
    currentGloss: document.getElementById('current-gloss'),
    textInput: document.getElementById('text-input'),
    btnTranslate: document.getElementById('btn-translate'),
    btnPlay: document.getElementById('btn-play'),
    btnStop: document.getElementById('btn-stop'),
    counter: document.getElementById('frame-counter'),
    glossDisplay: document.getElementById('glosses-display'),
    errorMsg: document.getElementById('error-msg'),
    vocabWords: document.getElementById('vocab-words'),
    vocabPanel: document.getElementById('vocab-panel'),
    videoPanel: document.getElementById('video-panel'),
    refVideo: document.getElementById('ref-video'),
    refVideoStatus: document.getElementById('ref-video-status')
};

// ═══════════════════════════════════════════════════════════════════════════════
// ENHANCED 3D HUMANOID MANNEQUIN
// ═══════════════════════════════════════════════════════════════════════════════

const SCALE = 1.6;
const OFFSET_Y = 0.8;
let AVATAR_STYLE = 'D'; // 'A' = Professional Male, 'C' = Cyborg, 'D' = Elder Mentor

function mp2three(lm) {
    return new THREE.Vector3(
        (lm.x - 0.5) * SCALE,
        -(lm.y - 0.5) * SCALE + OFFSET_Y,
        -(lm.z || 0) * SCALE
    );
}

// ─── MATERIALS ───────────────────────────────────────────────────────────────
const eyePupilMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
const shoeMat = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.7 });

// Initialize base materials immediately to avoid ReferenceError in LIMBS/JOINTS
let skinMat = new THREE.MeshPhysicalMaterial({ color: 0xd4a574, roughness: 0.7 });
let skinDarkMat = new THREE.MeshPhysicalMaterial({ color: 0xc49060, roughness: 0.7 });
let shirtMat = new THREE.MeshPhysicalMaterial({ color: 0x2563eb, roughness: 0.65 });
let pantsMat = new THREE.MeshPhysicalMaterial({ color: 0x1e293b, roughness: 0.8 });
let hairMat = new THREE.MeshPhysicalMaterial({ color: 0x2c1810, roughness: 0.85 });
let lipMat = new THREE.MeshPhysicalMaterial({ color: 0xb8785a, roughness: 0.5 });
let fingerMat = new THREE.MeshPhysicalMaterial({ color: 0xd4a574, roughness: 0.55 });
let eyeIrisMat = new THREE.MeshStandardMaterial({ color: 0x3b2314 });
let eyeWhiteMat = new THREE.MeshStandardMaterial({ color: 0xffffff });

function updateMaterials() {
    console.log("Updating materials for style:", AVATAR_STYLE);
    if (AVATAR_STYLE === 'C') {
        skinMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 0.15, metalness: 0.1 });
        skinDarkMat = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.2 });
        shirtMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.5 });
        pantsMat = shirtMat;
        hairMat = skinMat;
        lipMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
        fingerMat = skinMat;
    } else if (AVATAR_STYLE === 'D') {
        skinMat = new THREE.MeshPhysicalMaterial({ color: 0xd09f6e, roughness: 0.75, metalness: 0.0, clearcoat: 0.05, clearcoatRoughness: 0.5 });
        skinDarkMat = new THREE.MeshPhysicalMaterial({ color: 0xbe8858, roughness: 0.75, metalness: 0.0 });
        shirtMat = new THREE.MeshPhysicalMaterial({ color: 0x5a2035, roughness: 0.8, metalness: 0.0 });
        pantsMat = new THREE.MeshPhysicalMaterial({ color: 0x222222, roughness: 0.8, metalness: 0.0 });
        hairMat = new THREE.MeshPhysicalMaterial({ color: 0xb0b0b0, roughness: 0.9, metalness: 0.0, clearcoat: 0.1 });
        lipMat = new THREE.MeshPhysicalMaterial({ color: 0xac6f52, roughness: 0.6, clearcoat: 0.1 });
        fingerMat = new THREE.MeshPhysicalMaterial({ color: 0xd09f6e, roughness: 0.6, metalness: 0.0 });
    } else {
        skinMat = new THREE.MeshPhysicalMaterial({ color: 0xd4a574, roughness: 0.7, metalness: 0.0, clearcoat: 0.05, clearcoatRoughness: 0.4 });
        skinDarkMat = new THREE.MeshPhysicalMaterial({ color: 0xc49060, roughness: 0.7, metalness: 0.0 });
        shirtMat = new THREE.MeshPhysicalMaterial({ color: 0x2563eb, roughness: 0.65, metalness: 0.0 });
        pantsMat = new THREE.MeshPhysicalMaterial({ color: 0x1e293b, roughness: 0.8, metalness: 0.0 });
        hairMat = new THREE.MeshPhysicalMaterial({ color: 0x2c1810, roughness: 0.85, metalness: 0.0, clearcoat: 0.3 });
        lipMat = new THREE.MeshPhysicalMaterial({ color: 0xb8785a, roughness: 0.5, clearcoat: 0.15 });
        fingerMat = new THREE.MeshPhysicalMaterial({ color: 0xd4a574, roughness: 0.55, metalness: 0.0 });
    }

    eyeIrisMat.color.set(AVATAR_STYLE === 'C' ? 0x00ffff : (AVATAR_STYLE === 'D' ? 0x4a4a4a : 0x3b2314));
    eyeIrisMat.emissive.set(AVATAR_STYLE === 'C' ? 0x00ffff : 0x000000);
    eyeIrisMat.emissiveIntensity = (AVATAR_STYLE === 'C' ? 2 : 0);

    // CRITICAL: Re-map materials in the definition arrays so new meshes pick them up
    LIMBS.forEach(l => {
        if (l.id.includes("torso") || l.id.includes("upper_arm") || l.id.includes("shoulders")) l.mat = shirtMat;
        else if (l.id.includes("leg") || l.id === "hips_bar") l.mat = pantsMat;
        else if (l.id.includes("lower_arm")) l.mat = skinMat;
        else if (l.id.includes("foot")) l.mat = shoeMat;
    });
    JOINTS.forEach(j => {
        if (j.id.includes("shoulder") || j.id.includes("hip")) j.mat = (j.id.includes("hip") ? pantsMat : shirtMat);
        else if (j.id.includes("elbow") || j.id.includes("wrist")) j.mat = skinMat;
        else if (j.id.includes("ankle")) j.mat = shoeMat;
    });
}

// ─── MANNEQUIN STORAGE ───────────────────────────────────────────────────────
const M = {
    joints: {},       // Sphere meshes at joints
    limbs: {},        // Capsule/cylinder meshes connecting joints
    head: null,       // Head group
    torsoFill: null,  // Torso fill mesh
    rFingers: [],     // Right hand finger segments
    lFingers: [],     // Left hand finger segments
    // Dynamic face parts
    leftEyelid: null,
    rightEyelid: null,
    jaw: null,
    leftEyebrow: null,
    rightEyebrow: null,
    leftIris: null,
    rightIris: null,
    rFingerJoints: [],
    lFingerJoints: [],
    neckMesh: null,
};

// Limb definitions: [id, poseIdxA, poseIdxB, radius, material]
const LIMBS = [
    // Torso edges
    { id: "torso_r", a: 12, b: 24, r: 0.055, mat: shirtMat },
    { id: "torso_l", a: 11, b: 23, r: 0.055, mat: shirtMat },
    { id: "shoulders", a: 11, b: 12, r: 0.045, mat: shirtMat },
    { id: "hips_bar", a: 23, b: 24, r: 0.045, mat: pantsMat },
    // Arms
    { id: "r_upper_arm", a: 12, b: 14, r: 0.038, mat: shirtMat },
    { id: "r_lower_arm", a: 14, b: 16, r: 0.035, mat: skinMat },
    { id: "l_upper_arm", a: 11, b: 13, r: 0.038, mat: shirtMat },
    { id: "l_lower_arm", a: 13, b: 15, r: 0.035, mat: skinMat },
    // Legs
    { id: "r_upper_leg", a: 24, b: 26, r: 0.050, mat: pantsMat },
    { id: "r_lower_leg", a: 26, b: 28, r: 0.040, mat: pantsMat },
    { id: "l_upper_leg", a: 23, b: 25, r: 0.050, mat: pantsMat },
    { id: "l_lower_leg", a: 25, b: 27, r: 0.040, mat: pantsMat },
    // Feet
    { id: "r_foot", a: 28, b: 32, r: 0.030, mat: shoeMat },
    { id: "l_foot", a: 27, b: 31, r: 0.030, mat: shoeMat },
];

// Joint sphere definitions
const JOINTS = [
    { id: "r_shoulder", idx: 12, r: 0.045, mat: shirtMat },
    { id: "l_shoulder", idx: 11, r: 0.045, mat: shirtMat },
    { id: "r_elbow", idx: 14, r: 0.038, mat: skinMat },
    { id: "l_elbow", idx: 13, r: 0.038, mat: skinMat },
    { id: "r_wrist", idx: 16, r: 0.052, mat: skinMat },
    { id: "l_wrist", idx: 15, r: 0.052, mat: skinMat },
    { id: "r_hip", idx: 24, r: 0.045, mat: pantsMat },
    { id: "l_hip", idx: 23, r: 0.045, mat: pantsMat },
    { id: "r_knee", idx: 26, r: 0.038, mat: pantsMat },
    { id: "l_knee", idx: 25, r: 0.038, mat: pantsMat },
    { id: "r_ankle", idx: 28, r: 0.032, mat: shoeMat },
    { id: "l_ankle", idx: 27, r: 0.032, mat: shoeMat },
];

function buildMannequin() {
    updateMaterials();

    // Joint spheres
    JOINTS.forEach(j => {
        const mesh = new THREE.Mesh(new THREE.SphereGeometry(j.r, 16, 16), j.mat);
        mesh.name = "joint_" + j.id;
        mesh.castShadow = true;
        mesh.visible = false; // MUST HIDE
        mannequinGroup.add(mesh);
        M.joints[j.id] = mesh;
    });

    // Limb capsules
    LIMBS.forEach(l => {
        const geo = new THREE.CylinderGeometry(l.r, l.r * 0.9, 1, 12, 1);
        const mesh = new THREE.Mesh(geo, l.mat);
        mesh.name = "limb_" + l.id;
        mesh.castShadow = true;
        mesh.visible = false; // MUST HIDE until positioned
        mannequinGroup.add(mesh);
        M.limbs[l.id] = mesh;
    });

    // Head Integrated Base (Archetype branching)
    const headGroup = new THREE.Group();

    if (AVATAR_STYLE === 'C') {
        // Option C: Cyborg Pill-Head
        const skull = new THREE.Mesh(
            new THREE.SphereGeometry(0.10, 32, 24),
            skinMat
        );
        skull.scale.set(1.0, 1.25, 1.0);
        headGroup.add(skull);
    } else {
        // Option A: Human Sculpt
        const skull = new THREE.Mesh(
            new THREE.SphereGeometry(0.10, 32, 24),
            skinMat
        );
        skull.scale.set(1.0, 1.1, 0.8);
        headGroup.add(skull);

        // Integrated Lower Face (Jaw/Chin Block)
        const lowerFace = new THREE.Mesh(
            new THREE.BoxGeometry(0.17, 0.12, 0.08),
            skinMat
        );
        lowerFace.position.set(0, -0.04, 0.04);
        headGroup.add(lowerFace);

        // Chiseled Cheekbones (Integrated)
        [-0.075, 0.075].forEach(xOff => {
            const cheek = new THREE.Mesh(
                new THREE.SphereGeometry(0.035, 16, 16),
                skinMat
            );
            cheek.position.set(xOff, -0.01, 0.06);
            cheek.scale.set(1.1, 1.4, 0.5);
            headGroup.add(cheek);
        });

        // Short cropped hair cap
        const hairCap = new THREE.Mesh(
            new THREE.SphereGeometry(0.106, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.58),
            hairMat
        );
        hairCap.position.y += 0.02;
        headGroup.add(hairCap);
    }

    // Sideburns / Temple hair (Conditional)
    if (AVATAR_STYLE !== 'C') {
        [-0.085, 0.085].forEach(xOff => {
            const temple = new THREE.Mesh(
                new THREE.BoxGeometry(0.015, 0.06, 0.04),
                hairMat
            );
            temple.position.set(xOff, 0.02, 0.0);
            headGroup.add(temple);
        });
    }

    // ─── EYES (Archetype branching) ──────────────────────────────────
    const eyePositions = [{ x: -0.032, id: 'left' }, { x: 0.032, id: 'right' }];
    eyePositions.forEach(ep => {
        if (AVATAR_STYLE === 'C') {
            // Option C: Glowing rectangle panels
            const panel = new THREE.Mesh(
                new THREE.BoxGeometry(0.045, 0.008, 0.01),
                eyeIrisMat
            );
            panel.position.set(ep.x, 0.02, 0.100);
            headGroup.add(panel);
            if (ep.id === 'left') M.leftIris = panel;
            else M.rightIris = panel;
        } else {
            // Option A: Human Almond
            const eyeWhite = new THREE.Mesh(
                new THREE.SphereGeometry(0.014, 16, 16),
                eyeWhiteMat
            );
            eyeWhite.position.set(ep.x, 0.018, 0.095);
            eyeWhite.scale.set(1.4, 0.7, 0.3);
            headGroup.add(eyeWhite);

            const iris = new THREE.Mesh(
                new THREE.SphereGeometry(0.008, 12, 12),
                eyeIrisMat
            );
            iris.position.set(ep.x, 0.018, 0.100);
            headGroup.add(iris);
            if (ep.id === 'left') M.leftIris = iris;
            else M.rightIris = iris;

            const pupil = new THREE.Mesh(
                new THREE.SphereGeometry(0.003, 10, 10),
                new THREE.MeshStandardMaterial({ color: 0x000000 })
            );
            pupil.position.set(ep.x, 0.018, 0.117);
            headGroup.add(pupil);

            // Eyelid (Integrated surface)
            const eyelid = new THREE.Mesh(
                new THREE.SphereGeometry(0.016, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.45),
                skinMat
            );
            eyelid.position.set(ep.x, 0.024, 0.100);
            headGroup.add(eyelid);
            if (ep.id === 'left') M.leftEyelid = eyelid;
            else M.rightEyelid = eyelid;
        }
    });

    // ─── EYEBROWS (Conditional) ──────────────────────────────────
    if (AVATAR_STYLE !== 'C') {
        const browMat = new THREE.MeshStandardMaterial({ color: AVATAR_STYLE === 'D' ? 0x8a8a8a : 0x2c1810, roughness: 0.9 });
        [{ x: -0.035, id: 'left' }, { x: 0.035, id: 'right' }].forEach(ep => {
            const brow = new THREE.Mesh(
                new THREE.BoxGeometry(0.05, 0.015, 0.01),
                browMat
            );
            brow.position.set(ep.x, 0.045, 0.110);
            brow.rotation.z = ep.id === 'left' ? 0.2 : -0.2;
            headGroup.add(brow);
            if (ep.id === 'left') M.leftEyebrow = brow;
            else M.rightEyebrow = brow;
        });
    }

    // ─── NOSE (Conditional) ────────────────────────────────────────────
    if (AVATAR_STYLE !== 'C') {
        const noseGeo = new THREE.SphereGeometry(0.016, 10, 10);
        const noseMesh = new THREE.Mesh(noseGeo, skinDarkMat);
        noseMesh.position.set(0, -0.005, 0.122);
        noseMesh.scale.set(0.85, 0.7, 0.85);
        headGroup.add(noseMesh);

        // Nose bridge
        const bridge = new THREE.Mesh(
            new THREE.CylinderGeometry(0.006, 0.008, 0.03, 6),
            skinMat
        );
        bridge.position.set(0, 0.012, 0.115);
        bridge.rotation.x = 0.15;
        headGroup.add(bridge);

        // Nostrils
        [-0.008, 0.008].forEach(xOff => {
            const nostril = new THREE.Mesh(
                new THREE.SphereGeometry(0.005, 6, 6),
                new THREE.MeshStandardMaterial({ color: 0x8a6040 })
            );
            nostril.position.set(xOff, -0.015, 0.118);
            headGroup.add(nostril);
        });
    }

    // ─── MOUTH / JAW (Archetype branching) ───────────────────────────
    if (AVATAR_STYLE === 'C') {
        // Option C minimalist mouth indicator
        const mouth = new THREE.Mesh(
            new THREE.BoxGeometry(0.035, 0.004, 0.005),
            new THREE.MeshStandardMaterial({ color: 0x444444 })
        );
        mouth.position.set(0, -0.05, 0.10);
        headGroup.add(mouth);
        M.jaw = mouth;
    } else {
        // Upper lip
        const upperLip = new THREE.Mesh(
            new THREE.SphereGeometry(0.016, 12, 8),
            lipMat
        );
        upperLip.position.set(0, -0.034, 0.110);
        upperLip.scale.set(1.4, 0.35, 0.45);
        headGroup.add(upperLip);

        // Smile corners (subtle upward curve)
        [-0.018, 0.018].forEach(xOff => {
            const corner = new THREE.Mesh(
                new THREE.SphereGeometry(0.005, 8, 8),
                lipMat
            );
            corner.position.set(xOff, -0.032, 0.108);
            headGroup.add(corner);
        });

        // Lower jaw (moves for mouth open/close)
        const jaw = new THREE.Mesh(
            new THREE.SphereGeometry(0.015, 12, 8),
            lipMat
        );
        jaw.position.set(0, -0.040, 0.108);
        jaw.scale.set(1.3, 0.3, 0.45);
        headGroup.add(jaw);
        M.jaw = jaw;

        // Mouth interior (dark)
        const mouthInner = new THREE.Mesh(
            new THREE.SphereGeometry(0.010, 8, 8),
            new THREE.MeshStandardMaterial({ color: 0x401010 })
        );
        mouthInner.position.set(0, -0.038, 0.102);
        mouthInner.scale.set(1.2, 0.3, 0.3);
        headGroup.add(mouthInner);
    }

    // Ears (Conditional)
    if (AVATAR_STYLE !== 'C') {
        [-0.092, 0.092].forEach(xOff => {
            const ear = new THREE.Mesh(
                new THREE.TorusGeometry(0.018, 0.006, 8, 12, Math.PI * 1.5),
                skinDarkMat
            );
            ear.position.set(xOff, 0.01, 0.0);
            ear.rotation.y = xOff > 0 ? -0.4 : 0.4;
            headGroup.add(ear);

            const innerEar = new THREE.Mesh(
                new THREE.SphereGeometry(0.012, 8, 8),
                skinMat
            );
            innerEar.position.set(xOff * 0.95, 0.01, 0.0);
            innerEar.scale.set(0.3, 0.7, 0.5);
            headGroup.add(innerEar);
        });
    }

    mannequinGroup.add(headGroup);
    M.head = headGroup;

    // ─── LIMBS & JOINTS ──────────────────────────────────────────────────
    // Neck and Torso Fill removed for stability in Phase 8 fix.
    M.neckMesh = null;
    M.torsoFill = null;

    // ─── FINGER MESHES ───────────────────────────────────────────────────
    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
        // Right hand
        const rGeo = new THREE.CylinderGeometry(0.007, 0.005, 1, 6);
        const rMesh = new THREE.Mesh(rGeo, fingerMat);
        rMesh.castShadow = true;
        rMesh.visible = false;
        mannequinGroup.add(rMesh);
        M.rFingers.push(rMesh);

        // Left hand
        const lGeo = new THREE.CylinderGeometry(0.007, 0.005, 1, 6);
        const lMesh = new THREE.Mesh(lGeo, fingerMat);
        lMesh.castShadow = true;
        lMesh.visible = false;
        mannequinGroup.add(lMesh);
        M.lFingers.push(lMesh);
    }

    // Finger joint spheres
    for (let i = 0; i < 21; i++) {
        const rGeo = new THREE.SphereGeometry(i === 0 ? 0.045 : 0.008, 8, 8);
        const rm = new THREE.Mesh(rGeo, fingerMat);
        rm.castShadow = true;
        rm.visible = false;
        mannequinGroup.add(rm);
        M.rFingerJoints.push(rm);

        const lGeo = new THREE.SphereGeometry(i === 0 ? 0.045 : 0.008, 8, 8);
        const lm = new THREE.Mesh(lGeo, fingerMat);
        lm.castShadow = true;
        lm.visible = false;
        mannequinGroup.add(lm);
        M.lFingerJoints.push(lm);
    }

    // HEAD Visibility
    if (M.head) M.head.visible = false;

    UI.status.innerText = "Avatar 3D Listo. Escribe texto para traducir.";
}

// ─── POSITION A CYLINDER BETWEEN TWO 3D POINTS ──────────────────────────────
function posLimb(mesh, pA, pB) {
    if (!pA || !pB) return;
    const dist = pA.distanceTo(pB);
    if (dist < 0.001 || dist > 2.0) { // Safety guard for massive or zero limbs
        mesh.visible = false;
        return;
    }
    mesh.visible = true;
    const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
    mesh.position.copy(mid);
    const dir = new THREE.Vector3().subVectors(pB, pA);
    const length = dir.length();
    mesh.scale.set(1, length, 1);
    const axis = new THREE.Vector3(0, 1, 0);
    mesh.quaternion.setFromUnitVectors(axis, dir.normalize());
}

// ─── UPDATE MANNEQUIN EVERY FRAME ────────────────────────────────────────────
function updateMannequin(frame) {
    const pose = frame.poseLandmarks;
    if (!pose || pose.length < 25) return;

    // Re-link disconnected tracking geometry (MediaPipe confidence recovery)
    // If hand landmarks exist, enforce that pose wrists snap to hand wrists
    if (frame.rightHandLandmarks && frame.rightHandLandmarks.length > 0 && pose[12]) {
        const hWrist = frame.rightHandLandmarks[0];
        pose[16] = { x: hWrist.x, y: hWrist.y, z: hWrist.z };
        // Estimate a natural elbow if the arm is folded
        pose[14].x = (pose[12].x + hWrist.x) / 2 + 0.05; // Bow out slightly
        pose[14].y = Math.max(pose[12].y, hWrist.y) + Math.abs(pose[12].x - hWrist.x) * 0.5; // Drop elbow down
    }
    if (frame.leftHandLandmarks && frame.leftHandLandmarks.length > 0 && pose[11]) {
        const hWrist = frame.leftHandLandmarks[0];
        pose[15] = { x: hWrist.x, y: hWrist.y, z: hWrist.z };
        // Estimate a natural elbow
        pose[13].x = (pose[11].x + hWrist.x) / 2 - 0.05; // Bow out locally
        pose[13].y = Math.max(pose[11].y, hWrist.y) + Math.abs(pose[11].x - hWrist.x) * 0.5;
    }

    const p = pose.map((lm, i) => {
        const vec = mp2three(lm);
        // Progressive Z-nudge to keep arms and hands strictly in front of the torso
        if (i === 13 || i === 14) vec.z += 0.05; // Elbows slightly forward
        if (i === 15 || i === 16) vec.z += 0.10; // Wrists pushed forward
        if (i >= 17 && i <= 22) vec.z += 0.10;   // Hand indices in pose
        return vec;
    });

    // Joints
    JOINTS.forEach(j => {
        const mesh = M.joints[j.id];
        if (mesh && p[j.idx]) mesh.position.copy(p[j.idx]);
    });

    // Limbs
    LIMBS.forEach(l => {
        const mesh = M.limbs[l.id];
        if (mesh && p[l.a] && p[l.b]) posLimb(mesh, p[l.a], p[l.b]);
    });

    // Head — anchor strictly relative to shoulders for humanoid torso integrity
    if (M.head && p[11] && p[12]) {
        const shMid = new THREE.Vector3().addVectors(p[11], p[12]).multiplyScalar(0.5);
        // Humanoid Constraint: Keep head at stable height above shoulders
        M.head.position.set(shMid.x, shMid.y + 0.12, shMid.z);
    }

    // ─── FACE ANIMATION from faceLandmarks ────────────────────────────────
    if (frame.faceLandmarks && frame.faceLandmarks.length > 10) {
        const fl = frame.faceLandmarks;

        // Mouth open/close: vertical distance between upper lip (13) and lower lip (14)
        // MediaPipe face mesh: 13=upper lip center, 14=lower lip center
        if (fl[13] && fl[14] && M.jaw) {
            const mouthOpen = Math.abs(fl[14].y - fl[13].y) * 8;
            const jawDrop = Math.min(0.015, mouthOpen * 0.015);
            M.jaw.position.y = -0.040 - jawDrop;
            M.jaw.scale.y = 0.3 + mouthOpen * 0.4;
        }

        // Eye blink: vertical distance between upper/lower eyelid landmarks
        // MediaPipe: Left eye top 159, bottom 145. Right eye top 386, bottom 374.
        if (fl[159] && fl[145] && M.leftEyelid) {
            const leftOpen = Math.abs(fl[159].y - fl[145].y) * 20;
            M.leftEyelid.rotation.x = -0.3 - Math.max(0, 1 - leftOpen) * 0.8;
        }
        if (fl[386] && fl[374] && M.rightEyelid) {
            const rightOpen = Math.abs(fl[386].y - fl[374].y) * 20;
            M.rightEyelid.rotation.x = -0.3 - Math.max(0, 1 - rightOpen) * 0.8;
        }

        // Eyebrow raise: compare brow landmarks to eye top
        // Left brow 66 vs eye 159, Right brow 296 vs eye 386
        if (fl[66] && fl[159] && M.leftEyebrow) {
            const browRaise = Math.abs(fl[66].y - fl[159].y) * 15;
            M.leftEyebrow.position.y = 0.045 + Math.min(0.01, browRaise * 0.01);
        }
        if (fl[296] && fl[386] && M.rightEyebrow) {
            const browRaise = Math.abs(fl[296].y - fl[386].y) * 15;
            M.rightEyebrow.position.y = 0.045 + Math.min(0.01, browRaise * 0.01);
        }
    }

    // Neck — strictly anchored (Optional cylinder removed for stability)
    if (M.neckMesh && p[11] && p[12] && M.head) {
        M.neckMesh.visible = false;
    }

    // Torso fill — removed for stability in Phase 7 fix
    if (M.torsoFill) {
        M.torsoFill.visible = false;
    }

    // Right hand fingers with Z-nudge synchronized to wrist
    if (frame.rightHandLandmarks && frame.rightHandLandmarks.length === 21) {
        const rh = frame.rightHandLandmarks.map(lm => {
            const p = mp2three(lm);
            p.z += 0.10; // Must match the wrist Z-nudge (0.10) to prevent detachment
            return p;
        });
        HAND_CONNECTIONS.forEach(([a, b], i) => {
            if (M.rFingers[i] && rh[a] && rh[b]) posLimb(M.rFingers[i], rh[a], rh[b]);
        });
        rh.forEach((pos, i) => {
            if (M.rFingerJoints[i]) M.rFingerJoints[i].position.copy(pos);
        });
    }

    // Left hand fingers with Z-nudge synchronized to wrist
    if (frame.leftHandLandmarks && frame.leftHandLandmarks.length === 21) {
        const lh = frame.leftHandLandmarks.map(lm => {
            const p = mp2three(lm);
            p.z += 0.10; // Must match the wrist Z-nudge (0.10) to prevent detachment
            return p;
        });
        HAND_CONNECTIONS.forEach(([a, b], i) => {
            if (M.lFingers[i] && lh[a] && lh[b]) posLimb(M.lFingers[i], lh[a], lh[b]);
        });
        lh.forEach((pos, i) => {
            if (M.lFingerJoints[i]) M.lFingerJoints[i].position.copy(pos);
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THREE.JS SETUP
// ═══════════════════════════════════════════════════════════════════════════════

clock = new THREE.Clock();
initThreeJS();
fetchVocabulary();
setupDraggable(document.getElementById('video-panel'), document.getElementById('video-drag-handle'));

function initThreeJS() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 1.4, 3.5);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.physicallyCorrectLights = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    const container = document.getElementById('avatar-container');
    container.appendChild(renderer.domElement);
    renderer.domElement.style.position = "absolute";
    renderer.domElement.style.top = "0";
    renderer.domElement.style.left = "0";
    renderer.domElement.style.zIndex = "2";
    renderer.domElement.id = "three-canvas";

    // Lighting for realistic skin rendering
    const mainLight = new THREE.DirectionalLight(0xfff5e6, 1.5);
    mainLight.position.set(2, 3, 4);
    mainLight.castShadow = true;
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0xb0c4ff, 0.5);
    fillLight.position.set(-2, 1, 2);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
    rimLight.position.set(0, 2, -3);
    scene.add(rimLight);

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    mannequinGroup = new THREE.Group();
    scene.add(mannequinGroup);
    buildMannequin();

    orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
    orbitControls.target.set(0, 0.9, 0);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.08;
    orbitControls.minDistance = 1.5;
    orbitControls.maxDistance = 6;
    orbitControls.update();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        resizeSkeletonCanvas();
    });

    resizeSkeletonCanvas();
    animate();
}

function resizeSkeletonCanvas() {
    skCanvas.width = window.innerWidth;
    skCanvas.height = window.innerHeight;
}

// ─── ANIMATION LOOP ──────────────────────────────────────────────────────────
function animate(time) {
    requestAnimationFrame(animate);

    if (isPlaying && motionData.length > 0 && (time - lastTime > 40)) {
        const frame = motionData[currentFrame];
        if (frame) {
            drawSkeleton(frame);
            if (overlayCanvas && UI.refVideo && UI.videoPanel.style.display !== "none") {
                drawSkeletonOverlay(frame);
                if (!UI.refVideo.paused) UI.refVideo.pause();
                UI.refVideo.currentTime = currentFrame / 30.0;
            }
            updateMannequin(frame);
            if (frame.gloss) {
                UI.currentGloss.innerText = frame.gloss;
                highlightActiveGloss(frame.gloss);
            }
        }
        currentFrame = (currentFrame + 1) % motionData.length;
        UI.counter.innerText = `Frame: ${currentFrame + 1} / ${motionData.length}`;
        lastTime = time;
    }

    orbitControls.update();
    renderer.render(scene, camera);
}

// ─── SKELETON DRAWING (2D Canvas) ────────────────────────────────────────────
function drawSkeleton(frame) {
    const W = skCanvas.width, H = skCanvas.height;
    ctx.clearRect(0, 0, W, H);
    if (frame.poseLandmarks) drawPoseSkeleton(frame.poseLandmarks, W, H);
    if (frame.faceLandmarks) drawFace(frame.faceLandmarks, W, H);
    if (frame.rightHandLandmarks) drawHand(frame.rightHandLandmarks, COLORS.rhand, W, H);
    if (frame.leftHandLandmarks) drawHand(frame.leftHandLandmarks, COLORS.lhand, W, H);
}

function drawSkeletonOverlay(frame) {
    if (!overlayCanvas) return;
    const Otx = overlayCanvas.getContext('2d');
    const W = overlayCanvas.width = overlayCanvas.offsetWidth;
    const H = overlayCanvas.height = overlayCanvas.offsetHeight;
    Otx.clearRect(0, 0, W, H);
    const m1 = (lm) => ({ x: lm.x * W, y: lm.y * H });
    function drawL(a, b, c) {
        if (!a || !b) return;
        const p1 = m1(a), p2 = m1(b);
        Otx.beginPath(); Otx.moveTo(p1.x, p1.y); Otx.lineTo(p2.x, p2.y);
        Otx.strokeStyle = c; Otx.lineWidth = 3; Otx.stroke();
    }
    function drawP(lms, c) {
        Otx.fillStyle = c;
        lms.forEach(p => { const pm = m1(p); Otx.beginPath(); Otx.arc(pm.x, pm.y, 3, 0, 2 * Math.PI); Otx.fill(); });
    }
    if (frame.poseLandmarks) {
        POSE_CONNECTIONS.forEach(([i, j]) => drawL(frame.poseLandmarks[i], frame.poseLandmarks[j], 'rgba(0,255,255,0.7)'));
        drawP(frame.poseLandmarks, 'white');
    }
    if (frame.rightHandLandmarks) {
        HAND_CONNECTIONS.forEach(([i, j]) => drawL(frame.rightHandLandmarks[i], frame.rightHandLandmarks[j], 'rgba(255,50,50,0.9)'));
        drawP(frame.rightHandLandmarks, 'white');
    }
    if (frame.leftHandLandmarks) {
        HAND_CONNECTIONS.forEach(([i, j]) => drawL(frame.leftHandLandmarks[i], frame.leftHandLandmarks[j], 'rgba(50,255,50,0.9)'));
        drawP(frame.leftHandLandmarks, 'white');
    }
}

function mapLM(lm, W, H, padX = 0.15, padY = 0.1) {
    return { x: (padX + lm.x * (1 - 2 * padX)) * W, y: (padY + lm.y * (1 - 2 * padY)) * H };
}

function drawPoseSkeleton(landmarks, W, H) {
    POSE_CONNECTIONS.forEach(([a, b]) => {
        const lmA = landmarks[a], lmB = landmarks[b];
        if (!lmA || !lmB) return;
        const pA = mapLM(lmA, W, H), pB = mapLM(lmB, W, H);
        ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = COLORS.body; ctx.lineWidth = 3; ctx.globalAlpha = 0.4; ctx.stroke(); ctx.globalAlpha = 1;
    });
    landmarks.forEach(lm => {
        if (!lm) return;
        const p = mapLM(lm, W, H);
        ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.fill();
    });
}

function drawHand(landmarks, color, W, H) {
    ctx.globalAlpha = 0.5;
    HAND_CONNECTIONS.forEach(([a, b]) => {
        const lmA = landmarks[a], lmB = landmarks[b];
        if (!lmA || !lmB) return;
        const pA = mapLM(lmA, W, H), pB = mapLM(lmB, W, H);
        ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
    });
    ctx.globalAlpha = 1;
}

function drawFace(landmarks, W, H) {
    ctx.fillStyle = COLORS.face; ctx.globalAlpha = 0.25;
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i]; if (!lm) continue;
        const p = mapLM(lm, W, H);
        ctx.beginPath(); ctx.arc(p.x, p.y, 1, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
}

// ─── TRANSLATION FLOW ────────────────────────────────────────────────────────
async function translate() {
    const text = UI.textInput.value.trim();
    if (!text) return;
    UI.btnTranslate.disabled = true;
    UI.btnTranslate.innerText = "Sintetizando LSC...";
    UI.errorMsg.style.display = "none";
    UI.glossDisplay.innerHTML = "";
    try {
        const resp = await fetch(`${API_BASE}/translate`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }), signal: AbortSignal.timeout(60000),
        });
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || "Error"); }
        const data = await resp.json();
        loadAndPlay(data.frames, data.glosses, text);
    } catch (err) {
        if (err.name === "AbortError" || err.message.toLowerCase().includes("fetch")) {
            await fallbackLocalJSON(text);
        } else { showError(err.message); }
    } finally {
        UI.btnTranslate.disabled = false;
        UI.btnTranslate.innerText = "🔄 Traducir y Animar";
    }
}

async function fallbackLocalJSON(text) {
    try {
        const resp = await fetch('./lsc_motion_dummy.json');
        if (!resp.ok) throw new Error("No local JSON");
        const data = await resp.json();
        loadAndPlay(data.frames || data, data.glosses || ["LSC"], text);
    } catch { showError("API offline y no hay datos locales."); }
}

function loadAndPlay(frames, glosses, originalText) {
    motionData = frames.map(f => ({
        ...f,
        poseLandmarks: sanitize(f.poseLandmarks),
        rightHandLandmarks: sanitize(f.rightHandLandmarks),
        leftHandLandmarks: sanitize(f.leftHandLandmarks),
        faceLandmarks: sanitize(f.faceLandmarks),
    }));
    currentFrame = 0; isPlaying = true;
    UI.btnPlay.innerText = "⏸ Pause";
    UI.status.innerText = `Reproduciendo: "${originalText}"`;
    UI.counter.innerText = `Frame: 1 / ${motionData.length}`;
    displayGlosses(glosses);
}

function sanitize(lms) {
    if (!lms || !Array.isArray(lms)) return null;
    return lms.map(p => ({ x: p.x || 0, y: p.y || 0, z: p.z || 0, visibility: 1.0 }));
}

// ─── UI HELPERS ──────────────────────────────────────────────────────────────
function displayGlosses(glosses) {
    UI.glossDisplay.innerHTML = '';
    glosses.forEach(g => {
        const tag = document.createElement('span');
        tag.className = 'gloss-tag'; tag.id = `gloss-${g}`; tag.innerText = g;
        UI.glossDisplay.appendChild(tag);
    });
}

function highlightActiveGloss(currentGloss) {
    document.querySelectorAll('.gloss-tag').forEach(el =>
        el.classList.toggle('active', el.innerText === currentGloss)
    );
    if (currentGloss) {
        UI.videoPanel.style.display = "block";
        const newSrc = `${API_BASE}/api/video/${currentGloss}`;
        if (!UI.refVideo.src.endsWith(newSrc)) {
            UI.refVideo.src = newSrc;
            UI.refVideoStatus.innerText = `Mostrando: ${currentGloss}`;
            UI.refVideo.onerror = () => { UI.refVideoStatus.innerText = `Sin video: ${currentGloss}`; };
        }
    } else { UI.videoPanel.style.display = "none"; }
}

function showError(msg) { UI.errorMsg.style.display = "block"; UI.errorMsg.innerText = `⚠️ ${msg}`; }

// ─── VOCABULARY ──────────────────────────────────────────────────────────────
async function fetchVocabulary() {
    try {
        const resp = await fetch(`${API_BASE}/vocabulary`, { signal: AbortSignal.timeout(3000) });
        if (!resp.ok) return;
        const data = await resp.json();
        renderVocab(data.vocabulary);
    } catch { renderVocab(["HOLA", "GRACIAS", "AYUDA", "AMIGO", "AMOR", "BIEN", "CASA", "FAMILIA", "FELIZ"]); }
}

function renderVocab(vocab) {
    if (!vocab?.length) return;
    UI.vocabWords.innerHTML = vocab.map(w => `<span class="vocab-word" onclick="insertWord('${w}')">${w}</span>`).join('');
    UI.vocabPanel.style.display = 'block';
}

window.insertWord = (word) => {
    const cur = UI.textInput.value.trim();
    UI.textInput.value = cur ? `${cur} ${word.toLowerCase()}` : word.toLowerCase();
};

function rebuildMannequin() {
    console.log("Rebuilding Mannequin for style:", AVATAR_STYLE);
    if (!mannequinGroup) return;

    // 1. Dispose existing resources to prevent memory leaks
    mannequinGroup.traverse(obj => {
        if (obj.isMesh) {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        }
    });

    // 2. Remove all children from group
    while (mannequinGroup.children.length > 0) {
        mannequinGroup.remove(mannequinGroup.children[0]);
    }

    // 3. Reset storage object M
    M.joints = {}; M.limbs = {}; M.head = null; M.torsoFill = null;
    M.rFingers = []; M.lFingers = []; M.leftEyelid = null; M.rightEyelid = null;
    M.jaw = null; M.leftEyebrow = null; M.rightEyebrow = null; M.leftIris = null;
    M.rightIris = null; M.rFingerJoints = []; M.lFingerJoints = []; M.neckMesh = null;

    // 4. Re-calculate materials and rebuild meshes within existing group
    updateMaterials();
    buildMannequin();
}

// ─── UI HANDLERS ─────────────────────────────────────────────────────────────
UI.btnTranslate.addEventListener('click', translate);
UI.btnPlay.addEventListener('click', () => { isPlaying = !isPlaying; UI.btnPlay.innerText = isPlaying ? "⏸ Pause" : "▶ Play"; });
UI.btnStop.addEventListener('click', () => { isPlaying = false; currentFrame = 0; UI.btnPlay.innerText = "▶ Play"; UI.counter.innerText = "Frame: 0 / —"; });

const styleSelector = document.getElementById('style-selector');
if (styleSelector) {
    styleSelector.addEventListener('change', (e) => {
        AVATAR_STYLE = e.target.value;
        rebuildMannequin();
    });
}

// ─── DRAG LOGIC ──────────────────────────────────────────────────────────────
function setupDraggable(panel, handle) {
    if (!panel || !handle) return;
    let isDragging = false, currX, currY, initX, initY, xOff = 0, yOff = 0;
    handle.addEventListener("mousedown", e => { initX = e.clientX - xOff; initY = e.clientY - yOff; if (e.target === handle) isDragging = true; });
    document.addEventListener("mouseup", () => { initX = currX; initY = currY; isDragging = false; });
    document.addEventListener("mousemove", e => {
        if (!isDragging) return; e.preventDefault();
        currX = e.clientX - initX; currY = e.clientY - initY; xOff = currX; yOff = currY;
        panel.style.transform = `translate(${currX}px, ${currY}px)`;
    });
}
