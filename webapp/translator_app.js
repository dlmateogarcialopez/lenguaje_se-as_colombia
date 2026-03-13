// translator_app.js — LSC Translator Frontend Engine v2.1
// DIRECT IK ENGINE: Computes bone rotations directly from MediaPipe landmark
// vectors instead of relying on Kalidokit.Pose.solve which needs 3D world-space
// input but receives 2D image-space data (causing all signs to look the same).
//
// Architecture:
//  - Body (33 landmarks): Direct vector IK → shoulder, elbow, wrist, spine angles
//  - Hands (21 landmarks each): Kalidokit.Hand.solve (works great with 2D data)
//  - Face (468 landmarks): Kalidokit.Face.solve (works great with 2D data)

// ─── GLOBALS ─────────────────────────────────────────────────────────────────
let scene, camera, renderer, orbitControls, currentVrm, clock;
let isPlaying = false;
let currentFrame = 0;
let motionData = [];
let lastTime = 0;
const skCanvas = document.getElementById('skeleton-canvas');
const ctx = skCanvas.getContext('2d');

const COLORS = {
    body: '#00EDDA',
    rhand: '#ff6b6b',
    lhand: '#6bcbff',
    face: '#c0f566',
    spine: '#ffd700',
    joints: '#fff',
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
const boneCache = new Map();
const API_BASE = "";

// UI refs
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
};

clock = new THREE.Clock();
initThreeJS();
loadVRM("https://cdn.jsdelivr.net/gh/pixiv/three-vrm@dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm");
fetchVocabulary();

// ─── THREE.JS SETUP ──────────────────────────────────────────────────────────
function initThreeJS() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 1.3, 3.5);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    const container = document.getElementById('avatar-container');
    container.appendChild(renderer.domElement);
    renderer.domElement.style.position = "absolute";
    renderer.domElement.style.zIndex = "2";

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));

    orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
    orbitControls.target.set(0, 0.9, 0);
    orbitControls.enableDamping = true;
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

function loadVRM(url) {
    UI.status.innerText = "Cargando Avatar VRM...";
    const loader = new THREE.GLTFLoader();
    loader.register((parser) => new THREE.VRMLoaderPlugin(parser));
    loader.load(url, (gltf) => {
        const vrm = gltf.userData.vrm;
        THREE.VRMUtils.removeUnnecessaryVertices(gltf.scene);
        THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);
        scene.add(vrm.scene);
        currentVrm = vrm;
        // Rotate model 180° to face the camera (which is at +Z)
        // VRMUtils.rotateVRM0 permanently rotates the mesh if available
        if (THREE.VRMUtils?.rotateVRM0) {
            THREE.VRMUtils.rotateVRM0(vrm);
        } else {
            vrm.scene.rotation.y = Math.PI;
        }
        vrm.scene.position.set(0, -0.5, 0);
        UI.status.innerText = "Avatar Listo. Escribe texto para traducir.";
    }, null, (err) => {
        console.error(err);
        UI.status.innerText = "Error cargando avatar.";
    });
}

// ─── ANIMATION LOOP ──────────────────────────────────────────────────────────
function animate(time) {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();

    if (isPlaying && motionData.length > 0 && (time - lastTime > 40)) {
        const frame = motionData[currentFrame];
        if (frame) {
            drawSkeleton(frame);
            animateRig(frame);
            if (frame.gloss) {
                UI.currentGloss.innerText = frame.gloss;
                highlightActiveGloss(frame.gloss);
            }
        }
        currentFrame = (currentFrame + 1) % motionData.length;
        UI.counter.innerText = `Frame: ${currentFrame + 1} / ${motionData.length}`;
        lastTime = time;
    }

    if (currentVrm) currentVrm.update(delta);
    orbitControls.update();
    // renderer.render(scene, camera); // DISABLED 3D AVATAR AS REQUESTED
}

// ─── SKELETON DRAWING ─────────────────────────────────────────────────────────
function drawSkeleton(frame) {
    const W = skCanvas.width;
    const H = skCanvas.height;
    ctx.clearRect(0, 0, W, H);

    if (frame.poseLandmarks) drawPoseSkeleton(frame.poseLandmarks, W, H);
    if (frame.faceLandmarks) drawFace(frame.faceLandmarks, W, H);
    if (frame.rightHandLandmarks) drawHand(frame.rightHandLandmarks, COLORS.rhand, W, H);
    if (frame.leftHandLandmarks) drawHand(frame.leftHandLandmarks, COLORS.lhand, W, H);
}

function mapLM(lm, W, H, padX = 0.2, padY = 0.1) {
    // Normal LSC Translator view usually needs the avatar/skeleton centered
    // Landmarks are [0,1]. We scale and center.
    const x = (padX + lm.x * (1 - 2 * padX)) * W;
    const y = (padY + lm.y * (1 - 2 * padY)) * H;
    return { x, y };
}

function drawPoseSkeleton(landmarks, W, H) {
    POSE_CONNECTIONS.forEach(([a, b]) => {
        const lmA = landmarks[a], lmB = landmarks[b];
        if (!lmA || !lmB) return;
        const pA = mapLM(lmA, W, H);
        const pB = mapLM(lmB, W, H);
        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = COLORS.body;
        ctx.lineWidth = 4;
        ctx.globalAlpha = 0.6;
        ctx.stroke();
    });
    ctx.globalAlpha = 1;

    landmarks.forEach((lm, i) => {
        if (!lm || i > 24) return; // Only main body
        const p = mapLM(lm, W, H);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.joints;
        ctx.fill();
    });
}

function drawHand(landmarks, color, W, H) {
    HAND_CONNECTIONS.forEach(([a, b]) => {
        const lmA = landmarks[a], lmB = landmarks[b];
        if (!lmA || !lmB) return;
        const pA = mapLM(lmA, W, H);
        const pB = mapLM(lmB, W, H);
        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
    });
    landmarks.forEach(lm => {
        if (!lm) return;
        const p = mapLM(lm, W, H);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();
    });
}

function drawFace(landmarks, W, H) {
    ctx.fillStyle = COLORS.face;
    ctx.globalAlpha = 0.4;
    // Draw only a subset of face landmarks for performance and clarity
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        if (!lm) continue;
        const p = mapLM(lm, W, H);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DIRECT IK ENGINE
// ═══════════════════════════════════════════════════════════════════════════════
// WHY: MediaPipe body CSVs are 2D image-space (x,y ∈ [0,1], z ≈ 0).
// Kalidokit.Pose.solve needs 3D world-space → gives identical output for all signs.
// FIX: Compute bone rotations directly from landmark VECTORS using atan2 geometry.
//
// MediaPipe Pose landmark indices:
//   0=nose   11=L_shoulder  12=R_shoulder
//   13=L_elbow  14=R_elbow  15=L_wrist  16=R_wrist
//   23=L_hip    24=R_hip

function lm(landmarks, idx) {
    const p = landmarks[idx];
    return p ? { x: p.x || 0, y: p.y || 0, z: p.z || 0 } : null;
}

function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }; }
function len(v) { return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) || 1e-6; }
function norm(v) { const l = len(v); return { x: v.x / l, y: v.y / l, z: v.z / l }; }

/**
 * Compute VRM bone rotations from MediaPipe 2D image-space arm landmarks.
 *
 * VRM T-pose coordinate system (right arm):
 *   - ARM is extended to the RIGHT (+X world) at shoulder height
 *   - rightUpperArm.z: arm elevation (+z raises arm up, -z lowers below shoulder)
 *   - rightUpperArm.x: arm forward/back (negative = forward, positive = back)
 *   - rightLowerArm.z: elbow flex (negative = bent, 0 = straight)
 *
 * MediaPipe 2D image-space:
 *   - y increases DOWNWARD (top=0, bottom=1)
 *   - x increases to the RIGHT
 *
 * Key mappings:
 *   - arm raised UP   → elbow.y < shoulder.y → ua.y < 0 → z rotation POSITIVE
 *   - arm down        → elbow.y > shoulder.y → ua.y > 0 → z rotation NEGATIVE
 *   - arm forward     → z diff (z is small but nonzero in MediaPipe)
 *   - elbow bent      → lower-arm deviates from upper-arm direction
 */
function computeArmAngles(shoulder, elbow, wrist, side) {
    const ua = sub(elbow, shoulder);  // upper-arm direction vector
    const la = sub(wrist, elbow);    // lower-arm direction vector

    // --- Upper Arm ---
    // In image space: ua.y > 0 means elbow is BELOW shoulder (arm angled down)
    // VRM Z rotation of right arm: positive = arm up, negative = arm down
    // The neutral "arm at side" is roughly Z = -Math.PI/2 from T-pose
    // So: elevation = -(ua.y - 0) * factor, shifted to center around -PI/2
    const armLen = Math.sqrt(ua.x * ua.x + ua.y * ua.y) || 0.01;

    // Elevation angle from horizontal (0 = horizontal, +π/2 = up, -π/2 = down)
    const elevAngle = Math.atan2(-ua.y, armLen * 0.5);

    // For RIGHT arm: T-pose is arms-out-right. To hang arm down, need Z = -π/2.
    // elevAngle: -π/2 = hanging, 0 = horizontal, +π/2 = raised
    // VRM rightUpperArm.z = elevAngle (after accounting for T-pose)
    const upperArmZ = (side === 'right')
        ? -(Math.PI * 0.5 - elevAngle)  // -90 deg + elevation
        : (Math.PI * 0.5 - elevAngle);  // mirror for left

    // Arm swing forward/back from Z depth
    const upperArmX = Math.atan2(ua.z || 0, armLen) * 2.5;

    // Arm horizontal sweep (abduction/adduction)
    const sweep = Math.atan2(ua.x * (side === 'right' ? 1 : -1), armLen * 0.5);
    const upperArmY = sweep * 0.5;

    // --- Lower Arm (Elbow) ---
    // Compute angle between upper-arm and lower-arm in image space
    const uaN = norm(ua);
    const laN = norm(la);
    const dot = Math.max(-1, Math.min(1, uaN.x * laN.x + uaN.y * laN.y));
    const elbowAngle = Math.acos(dot);  // 0 = straight, π = fully bent

    // VRM elbow: negative Z = flex (bend) for right arm
    const lowerArmZ = (side === 'right') ? -elbowAngle * 0.8 : elbowAngle * 0.8;

    // Wrist pronation/supination from x-deviation of lower arm vs upper arm
    const pronate = (la.x - ua.x) * (side === 'right' ? -2 : 2);
    const lowerArmY = Math.max(-1.2, Math.min(1.2, pronate));

    return {
        upperArm: { x: upperArmX, y: upperArmY, z: upperArmZ },
        lowerArm: { x: 0, y: lowerArmY, z: lowerArmZ },
    };
}


function animateRig(frame) {
    if (!currentVrm || !frame) return;

    const landmarks = frame.poseLandmarks;

    // ── 1. BODY: DIRECT IK FROM LANDMARKS ──────────────────────────────────
    if (landmarks && landmarks.length >= 25) {
        const rSh = lm(landmarks, 12);
        const rEl = lm(landmarks, 14);
        const rWr = lm(landmarks, 16);
        const lSh = lm(landmarks, 11);
        const lEl = lm(landmarks, 13);
        const lWr = lm(landmarks, 15);
        const lHip = lm(landmarks, 23);
        const rHip = lm(landmarks, 24);
        const nose = lm(landmarks, 0);

        // Right arm IK
        if (rSh && rEl && rWr) {
            const r = computeArmAngles(rSh, rEl, rWr, 'right');
            rigBone("rightUpperArm", r.upperArm, 0.35);
            rigBone("rightLowerArm", r.lowerArm, 0.35);
        }

        // Left arm IK
        if (lSh && lEl && lWr) {
            const l = computeArmAngles(lSh, lEl, lWr, 'left');
            rigBone("leftUpperArm", l.upperArm, 0.35);
            rigBone("leftLowerArm", l.lowerArm, 0.35);
        }

        // Shoulder shrug: based on how high the shoulder is relative to hip
        if (rSh && rHip) {
            const dist = rSh.y - rHip.y;
            const shrug = Math.max(-0.4, Math.min(0.4, -(dist - 0.5) * 1.5));
            rigBone("rightShoulder", { x: 0, y: 0, z: shrug }, 0.25);
        }
        if (lSh && lHip) {
            const dist = lSh.y - lHip.y;
            const shrug = Math.max(-0.4, Math.min(0.4, -(dist - 0.5) * 1.5));
            rigBone("leftShoulder", { x: 0, y: 0, z: -shrug }, 0.25);
        }

        // Spine lean: based on shoulder-hip angle difference
        if (lSh && rSh && lHip && rHip) {
            const shoulderW = rSh.x - lSh.x;
            const hipW = rHip.x - lHip.x;
            const tiltZ = (shoulderW - hipW) * 0.8;
            const midShY = (lSh.y + rSh.y) / 2;
            const midHipY = (lHip.y + rHip.y) / 2;
            const tiltX = (midShY - midHipY - 0.45) * 0.5;
            rigBone("spine", { x: tiltX, y: 0, z: tiltZ * 0.6 }, 0.2);
            rigBone("upperChest", { x: tiltX, y: 0, z: tiltZ * 0.3 }, 0.2);
        }

        // Head / neck
        if (nose && lSh && rSh) {
            const midShX = (lSh.x + rSh.x) / 2;
            const midShY = (lSh.y + rSh.y) / 2;
            const neckYaw = (nose.x - midShX) * 1.5;
            const neckPitch = (nose.y - midShY - 0.3) * 0.8;
            rigBone("neck", { x: neckPitch, y: neckYaw, z: 0 }, 0.3);
            rigBone("head", { x: neckPitch * 0.4, y: neckYaw * 0.4, z: 0 }, 0.3);
        }
    }

    // ── 2. HANDS: Kalidokit works well with 2D hand data ───────────────────
    const rHandRig = frame.rightHandLandmarks
        ? Kalidokit.Hand.solve(frame.rightHandLandmarks, "Right") : null;
    const lHandRig = frame.leftHandLandmarks
        ? Kalidokit.Hand.solve(frame.leftHandLandmarks, "Left") : null;

    if (rHandRig) { rigHand(rHandRig, "right"); rigBone("rightHand", rHandRig.RightWrist, 0.4); }
    if (lHandRig) { rigHand(lHandRig, "left"); rigBone("leftHand", lHandRig.LeftWrist, 0.4); }

    // ── 3. FACE: Kalidokit works well with 2D face data ────────────────────
    const faceRig = frame.faceLandmarks
        ? Kalidokit.Face.solve(frame.faceLandmarks, { runtime: "mediapipe" }) : null;

    if (faceRig && currentVrm.expressionManager) {
        const em = currentVrm.expressionManager;
        const s = faceRig.mouth?.shape ?? {};
        em.setValue("aa", Math.max(0, s.A ?? 0));
        em.setValue("ee", Math.max(0, s.E ?? 0));
        em.setValue("ih", Math.max(0, s.I ?? 0));
        em.setValue("oh", Math.max(0, s.O ?? 0));
        em.setValue("ou", Math.max(0, s.U ?? 0));
        const eyeL = faceRig.eye?.l ?? 1, eyeR = faceRig.eye?.r ?? 1;
        em.setValue("blink", Math.max(0, 1 - (eyeL + eyeR) / 2));
        em.setValue("blinkLeft", Math.max(0, 1 - eyeL));
        em.setValue("blinkRight", Math.max(0, 1 - eyeR));
        if ((faceRig.brow ?? 0) > 0.6) em.setValue("surprised", 0.5);
    }
}

// ─── BONE APPLICATION ─────────────────────────────────────────────────────────
function rigBone(boneName, rot, lerp = 0.3) {
    if (!rot || isNaN(rot.x) || isNaN(rot.y) || isNaN(rot.z)) return;
    let bone = boneCache.get(boneName);
    if (!bone) {
        bone = currentVrm?.humanoid?.getNormalizedBoneNode?.(boneName)
            ?? currentVrm?.humanoid?.getBoneNode?.(boneName);
        if (bone) boneCache.set(boneName, bone);
    }
    if (!bone) return;
    const euler = new THREE.Euler(rot.x, rot.y, rot.z, 'XYZ');
    bone.quaternion.slerp(new THREE.Quaternion().setFromEuler(euler), lerp);
}

function rigHand(rig, side) {
    const p = side === "right" ? "right" : "left";
    const S = side.charAt(0).toUpperCase() + side.slice(1);
    ["Thumb", "Index", "Middle", "Ring", "Little"].forEach(finger => {
        ["Proximal", "Intermediate", "Distal"].forEach(joint => {
            const r = rig[`${S}${finger}${joint}`];
            if (r) rigBone(`${p}${finger}${joint}`, r, 0.5);
        });
    });
}

// Backwards-compat alias
function rigRotation(boneName, rot, damp = 1, lerpFactor = 0.3) {
    if (!rot) return;
    rigBone(boneName, { x: (rot.x ?? 0) * damp, y: (rot.y ?? 0) * damp, z: (rot.z ?? 0) * damp }, lerpFactor);
}
function rigPosition(boneName, pos, damp = 1, lerp = 0.3) {
    let bone = boneCache.get(boneName);
    if (!bone) {
        bone = currentVrm?.humanoid?.getNormalizedBoneNode?.(boneName)
            ?? currentVrm?.humanoid?.getBoneNode?.(boneName);
        if (bone) boneCache.set(boneName, bone);
    }
    if (!bone || !pos) return;
    bone.position.lerp(new THREE.Vector3(pos.x * damp, pos.y * damp, pos.z * damp), lerp);
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
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
            signal: AbortSignal.timeout(60000), // Increased to 60s for long dashboard phrases
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || "Error de traducción");
        }

        const data = await resp.json();
        loadAndPlay(data.frames, data.glosses, text);

    } catch (err) {
        if (err.name === "AbortError" || err.message.toLowerCase().includes("fetch")) {
            console.warn("API offline, cargando JSON local...");
            await fallbackLocalJSON(text);
        } else {
            showError(err.message);
        }
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
    } catch {
        showError("API offline y no hay datos locales. Ejecuta: python pipeline/lsc_api_server.py");
    }
}

function loadAndPlay(frames, glosses, originalText) {
    motionData = frames.map(frame => ({
        ...frame,
        poseLandmarks: sanitize(frame.poseLandmarks),
        rightHandLandmarks: sanitize(frame.rightHandLandmarks),
        leftHandLandmarks: sanitize(frame.leftHandLandmarks),
        faceLandmarks: sanitize(frame.faceLandmarks),
    }));

    currentFrame = 0;
    isPlaying = true;
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
        tag.className = 'gloss-tag';
        tag.id = `gloss-${g}`;
        tag.innerText = g;
        UI.glossDisplay.appendChild(tag);
    });
}

function highlightActiveGloss(currentGloss) {
    document.querySelectorAll('.gloss-tag').forEach(el =>
        el.classList.toggle('active', el.innerText === currentGloss)
    );
}

function showError(msg) {
    UI.errorMsg.style.display = "block";
    UI.errorMsg.innerText = `⚠️ ${msg}`;
}

// ─── VOCABULARY ──────────────────────────────────────────────────────────────
async function fetchVocabulary() {
    try {
        const resp = await fetch(`${API_BASE}/vocabulary`, { signal: AbortSignal.timeout(3000) });
        if (!resp.ok) return;
        const data = await resp.json();
        renderVocab(data.vocabulary);
    } catch {
        renderVocab(["HOLA", "GRACIAS", "AYUDA", "AMIGO", "AMOR", "BIEN", "CASA", "FAMILIA", "FELIZ", "MUJER", "PADRE", "MADRE"]);
    }
}

function renderVocab(vocab) {
    if (!vocab?.length) return;
    UI.vocabWords.innerHTML = vocab.map(w =>
        `<span class="vocab-word" onclick="insertWord('${w}')">${w}</span>`
    ).join('');
    UI.vocabPanel.style.display = 'block';
}

window.insertWord = (word) => {
    const cur = UI.textInput.value.trim();
    UI.textInput.value = cur ? `${cur} ${word.toLowerCase()}` : word.toLowerCase();
};

// ─── BUTTON EVENTS ───────────────────────────────────────────────────────────
UI.btnTranslate.addEventListener('click', translate);
UI.textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); translate(); }
});
UI.btnPlay.addEventListener('click', () => {
    isPlaying = !isPlaying;
    UI.btnPlay.innerText = isPlaying ? "⏸ Pause" : "▶ Play";
});
UI.btnStop.addEventListener('click', () => {
    isPlaying = false;
    currentFrame = 0;
    UI.btnPlay.innerText = "▶ Play";
    UI.counter.innerText = "Frame: 0 / —";
});
