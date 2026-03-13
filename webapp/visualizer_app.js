// visualizer_app.js — LSC Triple-Mode Visualizer v4.0 (3D Avatar)
// Panel 1: Skeleton canvas  — MediaPipe landmarks in real time
// Panel 2: Real signer video — original LSC50 dataset
// Panel 3: 3D Avatar (Three.js WebGL & RPM GLB) with Quaternion IK

'use strict';

// ═══ GLOBALS ═══════════════════════════════════════════════════════════════════
let motionData = [];
let currentFrame = 0;
let isPlaying = false;
let lastTime = 0;
const FRAME_MS = 40;   // ~25fps

// Canvas
const skCanvas = document.getElementById('skeleton-canvas');
const ctx = skCanvas.getContext('2d');

// UI
const UI = {
    status: document.getElementById('status-bar'),
    textInput: document.getElementById('text-input'),
    btnTranslate: document.getElementById('btn-translate'),
    btnPlay: document.getElementById('btn-play'),
    btnStop: document.getElementById('btn-stop'),
    modeToggle: document.getElementById('mode-toggle'),
    glossBox: document.getElementById('gloss-display'),
    vocabChips: document.getElementById('vocab-chips'),
    frameInfo: document.getElementById('frame-info'),
    errorMsg: document.getElementById('error-msg'),
    avatarPanel: document.getElementById('avatar-panel'),
};

// View mode: 'skeleton' | 'both' | 'video' | 'avatar'
let viewMode = 'skeleton';

// ═══ THREE.JS & AVATAR GLOBALS ═════════════════════════════════════════════════
let scene, camera, renderer, orbitControls;
let currentAvatar = null;
let avatarBones = {};
const headMesh = []; // For facial morphs

// ═══ SKELETON CANVAS COLORS ════════════════════════════════════════════════════
const COLORS = {
    body: '#00EDDA',
    rhand: '#ff6b6b',
    lhand: '#6bcbff',
    face: '#c0f566',
    spine: '#ffd700',
    joints: '#fff',
};

// MediaPipe Pose connections: [from, to]
const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [11, 12], [11, 23], [12, 24], [23, 24],
    [11, 13], [13, 15], [12, 14], [14, 16],
    [15, 17], [15, 19], [15, 21], [17, 19],
    [16, 18], [16, 20], [16, 22], [18, 20],
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// MediaPipe Hand connections (21 landmarks)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17],
];

// ═══ CANVAS SETUP & RESIZE ══════════════════════════════════════════════════════
function resizeCanvas() {
    const panel = document.getElementById('skeleton-panel');
    const w = panel.offsetWidth;
    const h = panel.offsetHeight;
    if (w > 0 && h > 0) {
        skCanvas.width = w;
        skCanvas.height = h;
    }
}

const skPanel = document.getElementById('skeleton-panel');
const ro = new ResizeObserver(() => resizeCanvas());
ro.observe(skPanel);
requestAnimationFrame(() => { resizeCanvas(); });
window.addEventListener('resize', resizeCanvas);

// ═══ SKELETON DRAW ENGINE ═══════════════════════════════════════════════════════
function drawSkeleton(frame) {
    const W = skCanvas.width;
    const H = skCanvas.height;
    ctx.clearRect(0, 0, W, H);

    const radGrad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.min(W, H) * 0.5);
    radGrad.addColorStop(0, 'rgba(0,237,218,0.04)');
    radGrad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = radGrad;
    ctx.fillRect(0, 0, W, H);

    if (frame.poseLandmarks && frame.poseLandmarks.length > 0) drawPoseSkeleton(frame.poseLandmarks, W, H);
    if (frame.faceLandmarks && frame.faceLandmarks.length > 0) drawFace(frame.faceLandmarks, W, H);
    if (frame.rightHandLandmarks && frame.rightHandLandmarks.length > 0) drawHand(frame.rightHandLandmarks, COLORS.rhand, W, H);
    if (frame.leftHandLandmarks && frame.leftHandLandmarks.length > 0) drawHand(frame.leftHandLandmarks, COLORS.lhand, W, H);

    if (frame.gloss) {
        ctx.font = 'bold 22px Outfit, sans-serif';
        ctx.fillStyle = COLORS.body;
        ctx.textAlign = 'center';
        ctx.globalAlpha = 0.85;
        ctx.fillText(frame.gloss, W / 2, H - 24);
        ctx.globalAlpha = 1;
    }
}

function mapLM(lm, W, H, padX = 0.08, padY = 0.05) {
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

        let color = COLORS.body;
        if ((a >= 11 && a <= 16) || (b >= 11 && b <= 16)) color = COLORS.body;
        if (a >= 23 || b >= 23) color = COLORS.spine;

        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.7;
        ctx.stroke();
        ctx.globalAlpha = 1;
    });

    landmarks.forEach((lm, i) => {
        if (!lm) return;
        const p = mapLM(lm, W, H);
        const isKeyJoint = [0, 11, 12, 13, 14, 15, 16, 23, 24].includes(i);
        const r = isKeyJoint ? 7 : 4;

        let col = COLORS.body;
        if (i <= 10) col = COLORS.face;
        if (i >= 23) col = COLORS.spine;

        ctx.shadowColor = col;
        ctx.shadowBlur = isKeyJoint ? 14 : 6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
        ctx.shadowBlur = 0;

        if (isKeyJoint) {
            const labels = {
                11: 'L.Sh', 12: 'R.Sh', 13: 'L.El', 14: 'R.El',
                15: 'L.Wr', 16: 'R.Wr', 0: 'Nariz', 23: 'L.Cad', 24: 'R.Cad'
            };
            if (labels[i]) {
                ctx.font = '10px Outfit, sans-serif';
                ctx.fillStyle = '#fff';
                ctx.globalAlpha = 0.7;
                ctx.textAlign = 'center';
                ctx.fillText(labels[i], p.x, p.y - 10);
                ctx.globalAlpha = 1;
            }
        }
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
        ctx.globalAlpha = 0.8;
        ctx.stroke();
        ctx.globalAlpha = 1;
    });

    landmarks.forEach((lm, i) => {
        if (!lm) return;
        const p = mapLM(lm, W, H);
        const r = i === 0 ? 6 : 3;
        ctx.shadowColor = color;
        ctx.shadowBlur = r === 6 ? 12 : 4;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.shadowBlur = 0;
    });
}

function drawFace(landmarks, W, H) {
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = COLORS.face;
    landmarks.forEach(lm => {
        if (!lm) return;
        const p = mapLM(lm, W, H);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
    });
    ctx.globalAlpha = 1;
}

// ═══ ANIMATION LOOP ═══════════════════════════════════════════════════════════
function animLoop(time) {
    requestAnimationFrame(animLoop);

    if (isPlaying && motionData.length > 0 && (time - lastTime > FRAME_MS)) {
        const frame = motionData[currentFrame];
        if (frame) {
            if (viewMode !== 'avatar' && viewMode !== 'video') drawSkeleton(frame);
            if (viewMode !== 'skeleton' && viewMode !== 'video') animateAvatar(frame);
            if (frame.gloss) highlightGloss(frame.gloss);
            UI.frameInfo.innerText = `Frame: ${currentFrame + 1} / ${motionData.length}`;
        }
        currentFrame = (currentFrame + 1) % motionData.length;
        lastTime = time;
    }
}
requestAnimationFrame(animLoop);

// ═══ CUSTOM ROBUST 3D KINEMATICS ENGINE (QUATERNION IK) ══════════════════════
function aimBone(bone, worldDirVector, defaultLocalDir = new THREE.Vector3(1, 0, 0)) {
    if (!bone || worldDirVector.lengthSq() < 0.0001) return;
    const parentRot = new THREE.Quaternion();
    if (bone.parent) bone.parent.getWorldQuaternion(parentRot);
    parentRot.invert();

    // Transform target world direction into the bone's local space
    const localTargetDir = worldDirVector.clone().applyQuaternion(parentRot).normalize();
    const qTarget = new THREE.Quaternion().setFromUnitVectors(defaultLocalDir, localTargetDir);
    bone.quaternion.slerp(qTarget, 0.85); // Smooth interpolation
    bone.updateMatrixWorld(true);
}

function animateAvatar(frame) {
    if (!currentAvatar || !frame.poseLandmarks) return;

    const pose = frame.poseLandmarks;
    // Map screen x,y coordinates to simple 3D vectors
    // MediaPipe: x is Left->Right, y is Top->Bottom.
    // ThreeJS/RPM: x is Right->Left (mirrored), y is Bottom->Top, z comes AT the camera.
    const P = i => pose[i] ? new THREE.Vector3(pose[i].x * 2 - 1, -(pose[i].y * 2 - 1), -pose[i].z) : null;

    const lSh = P(11), rSh = P(12), lEl = P(13), rEl = P(14), lWr = P(15), rWr = P(16);
    const nose = P(0);

    // --- Right Arm (+X is Right in ThreeJS/RPM rest pose) ---
    // Note: User's right is frame's left because MediaPipe is mirrored like a selfie. 
    // lSh is actually the user's right shoulder physically in video. 
    // We bind RightArm to lSh->lEl if we don't want to mirror, but wait: MediaPipe is already mirrored!
    // Left on screen = user's right arm. Right on screen = user's left arm.
    // Let's keep it simple: map R to R, L to L in screen space if we want a mirror avatar, which is best for viewing.

    // --- Right Arm (+X is Right in ThreeJS/RPM rest pose) ---
    // Note: User's right is frame's left because MediaPipe is mirrored like a selfie. 
    // Let's directly map Frame's Left to Avatar's Right Arm to act as a mirror
    if (lSh && lEl && avatarBones.RightArm) {
        aimBone(avatarBones.RightArm, new THREE.Vector3().subVectors(lEl, lSh).normalize(), new THREE.Vector3(1, 0, 0));
    }
    if (lEl && lWr && avatarBones.RightForeArm) {
        aimBone(avatarBones.RightForeArm, new THREE.Vector3().subVectors(lWr, lEl).normalize(), new THREE.Vector3(1, 0, 0));
    }

    // --- Left Arm (-X is Left in rest pose) ---
    // Frame's Right maps to Avatar's Left Arm
    if (rSh && rEl && avatarBones.LeftArm) {
        aimBone(avatarBones.LeftArm, new THREE.Vector3().subVectors(rEl, rSh).normalize(), new THREE.Vector3(-1, 0, 0));
    }
    if (rEl && rWr && avatarBones.LeftForeArm) {
        aimBone(avatarBones.LeftForeArm, new THREE.Vector3().subVectors(rWr, rEl).normalize(), new THREE.Vector3(-1, 0, 0));
    }

    // --- Head / Neck ---
    if (nose && rSh && lSh && avatarBones.Head) {
        const midSh = new THREE.Vector3().addVectors(rSh, lSh).multiplyScalar(0.5);
        const headDir = new THREE.Vector3().subVectors(nose, midSh).normalize();
        headDir.z -= 0.15; // Natural forward tilt
        aimBone(avatarBones.Head, headDir, new THREE.Vector3(0, 1, 0));
    }

    // --- Hands (Kalidokit 2D Fallback) ---
    if (frame.rightHandLandmarks && window.Kalidokit && avatarBones.LeftHand) {
        let rHandRig = Kalidokit.Hand.solve(frame.rightHandLandmarks, "Right");
        if (rHandRig) {
            avatarBones.LeftHand.rotation.set(rHandRig.RightWrist.x, rHandRig.RightWrist.y, rHandRig.RightWrist.z);
            avatarBones.LeftHand.updateMatrixWorld(true);
        }
    }
    if (frame.leftHandLandmarks && window.Kalidokit && avatarBones.RightHand) {
        let lHandRig = Kalidokit.Hand.solve(frame.leftHandLandmarks, "Left");
        if (lHandRig) {
            avatarBones.RightHand.rotation.set(lHandRig.LeftWrist.x, lHandRig.LeftWrist.y, lHandRig.LeftWrist.z);
            avatarBones.RightHand.updateMatrixWorld(true);
        }
    }

    // --- Face Details ---
    if (frame.faceLandmarks && headMesh.length > 0 && window.Kalidokit) {
        const faceRig = Kalidokit.Face.solve(frame.faceLandmarks, { runtime: "browser", video: null });
        if (faceRig) {
            headMesh.forEach(mesh => {
                if (!mesh.morphTargetDictionary) return;
                const setBlend = (name, val) => {
                    if (mesh.morphTargetDictionary[name] !== undefined) {
                        mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = val;
                    }
                };
                setBlend("mouthOpen", faceRig.mouth.y);
                setBlend("mouthSmile", faceRig.mouth.x);
                setBlend("eyeBlinkLeft", faceRig.eye.l);
                setBlend("eyeBlinkRight", faceRig.eye.r);
            });
        }
    }
}

// ═══ AVATAR (THREE.JS GLB) INIT ════════════════════════════════════════════════
function initVRM() {
    const container = document.getElementById('avatar-container');
    const loadingEl = document.getElementById('vrm-loading');
    if (!container) return;

    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    // Move camera significantly back and up to capture large/misaligned models
    camera.position.set(0, 1.5, 4.5);

    orbitControls = new THREE.OrbitControls(camera, renderer.domElement);
    // Target the center
    orbitControls.target.set(0, 1.0, 0);
    orbitControls.enablePan = false;
    orbitControls.enableZoom = true;

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);

    const loader = new THREE.GLTFLoader();
    // High quality realistic human model (Michelle) from official ThreeJS
    const url = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/Michelle.glb';

    loader.load(url, (gltf) => {
        currentAvatar = gltf.scene;
        scene.add(currentAvatar);

        currentAvatar.traverse(node => {
            if (node.isBone) {
                // Remove Mixamo prefix if present, and handle common Unity namings
                let name = node.name.replace('mixamorig', '');
                // Unity standard often just uses 'RightArm', 'RightForeArm', etc., which matches Mixamo after replace
                avatarBones[name] = node;
                console.log("Bone loaded:", name);
            }
            if (node.isMesh && node.morphTargetDictionary) {
                headMesh.push(node);
            }
            if (node.isMesh && node.material) {
                node.material.depthWrite = !node.material.transparent;
            }
        });

        // Debug output to see what we actually captured
        console.log("Avatar Bones mapped:", Object.keys(avatarBones).join(", "));

        // We removed the T-Pose artificial rotations because it was locking the arms locally and breaking the IK solver.
        // The quaternions should naturally align the vectors regardless of base pose.

        if (loadingEl) loadingEl.style.display = 'none';
        UI.status.innerText = 'Avatar Humano Listo. Escribe y traduce.';

        const animateThree = () => {
            requestAnimationFrame(animateThree);
            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        };
        animateThree();
    },
        (progress) => {
            if (progress.total > 0 && loadingEl) {
                UI.status.innerText = `Descargando Avatar 3D... ${Math.round((progress.loaded / progress.total) * 100)}%`;
            }
        },
        (err) => {
            console.error("No se pudo cargar GLB:", err);
            UI.status.innerText = 'Error cargando Avatar 3D. Verifica conexión a internet.';
            if (loadingEl) loadingEl.style.display = 'none';
        });

    const ro = new ResizeObserver(() => {
        if (!renderer || !camera || !container) return;
        renderer.setSize(container.clientWidth, container.clientHeight);
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
    });
    ro.observe(container);
}

// ═══ TRANSLATION ══════════════════════════════════════════════════════════════
async function translate() {
    const text = UI.textInput.value.trim();
    if (!text) return;
    UI.btnTranslate.disabled = true;
    UI.btnTranslate.innerText = 'Traduciendo...';
    showError('');

    try {
        const resp = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
            signal: AbortSignal.timeout(12000),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Error de traducción');
        const data = await resp.json();
        loadMotion(data.frames, data.glosses);
    } catch (err) {
        if (err.name === 'AbortError' || err.message.toLowerCase().includes('fetch')) {
            await fallbackJSON(text);
        } else {
            showError(err.message);
        }
    } finally {
        UI.btnTranslate.disabled = false;
        UI.btnTranslate.innerText = '🔄 Traducir';
    }
}

async function fallbackJSON(text) {
    try {
        const r = await fetch('./lsc_motion_dummy.json');
        if (!r.ok) throw new Error();
        const d = await r.json();
        loadMotion(d.frames || d, d.glosses || ['LSC']);
    } catch {
        showError('API offline. Ejecuta: python pipeline/lsc_api_server.py');
    }
}

function loadMotion(frames, glosses) {
    motionData = frames.map(f => ({
        ...f,
        poseLandmarks: sanitize(f.poseLandmarks),
        rightHandLandmarks: sanitize(f.rightHandLandmarks),
        leftHandLandmarks: sanitize(f.leftHandLandmarks),
        faceLandmarks: sanitize(f.faceLandmarks),
    }));
    currentFrame = 0;
    isPlaying = true;
    UI.btnPlay.innerText = '⏸ Pause';
    UI.status.innerText = `Reproduciendo ${motionData.length} frames — ${glosses.join(', ')}`;
    renderGlossChips(glosses);
}

function sanitize(lms) {
    if (!lms || !Array.isArray(lms)) return null;
    return lms.map(p => ({ x: p.x || 0, y: p.y || 0, z: p.z || 0, visibility: 1 }));
}

// ═══ VIDEO LOADING ═══════════════════════════════════════════════════════════
let currentVideoGloss = null;
const vidEl = document.getElementById('real-video');
const vidLoad = document.getElementById('video-loading');
const vidError = document.getElementById('video-error');
const vidPlch = document.getElementById('video-placeholder');
const vidBadge = document.getElementById('video-gloss-badge');

async function loadVideo(gloss) {
    if (!gloss || gloss === currentVideoGloss) return;
    currentVideoGloss = gloss;

    vidPlch.style.display = 'none';
    vidEl.style.display = 'none';
    vidError.style.display = 'none';
    vidLoad.style.display = 'flex';
    vidBadge.style.display = 'none';

    document.getElementById('video-panel-label').innerText = `🎥 Video Real — ${gloss}`;

    try {
        // Procedural signs starting with LETRA_ or NUMERO_ don't have videos
        if (gloss.startsWith('LETRA_') || gloss.startsWith('NUMERO_')) {
            vidLoad.style.display = 'none';
            vidPlch.style.display = 'flex';
            vidPlch.querySelector('.placeholder-text').innerText = 'Síntesis 3D Procedural (Sin Video)';
            return;
        }

        const resp = await fetch(`/api/video/${gloss}`, { method: 'HEAD', signal: AbortSignal.timeout(8000) });
        if (!resp.ok) throw new Error('not found');

        vidEl.src = `/api/video/${gloss}?t=${Date.now()}`;
        vidEl.style.display = 'block';
        vidLoad.style.display = 'none';
        vidBadge.innerText = gloss;
        vidBadge.style.display = 'block';
        await vidEl.play().catch(() => { });
    } catch {
        vidLoad.style.display = 'none';
        vidError.style.display = 'block';
        vidError.querySelector('span').innerText = `⚠️ Video de "${gloss}" no disponible`;
    }
}

// ═══ VOCABULARY ═══════════════════════════════════════════════════════════════
async function loadVocab() {
    try {
        const r = await fetch('/vocabulary', { signal: AbortSignal.timeout(3000) });
        if (!r.ok) return;
        const d = await r.json();
        renderVocab(d.vocabulary);
    } catch {
        renderVocab(['HOLA', 'AMIGO', 'GRACIAS', 'AYUDA', 'AMOR', 'BIEN', 'CASA', 'FAMILIA', 'FELIZ', 'PADRE', 'MADRE', 'BUENAS']);
    }
}

function renderVocab(words) {
    UI.vocabChips.innerHTML = words.map(w =>
        `<span class="vocab-chip" onclick="insertWord('${w}')">${w}</span>`
    ).join('');
}

window.insertWord = (w) => {
    const cur = UI.textInput.value.trim();
    UI.textInput.value = cur ? `${cur} ${w.toLowerCase()}` : w.toLowerCase();
    loadVideo(w.toUpperCase());
};

// ═══ GLOSS DISPLAY ════════════════════════════════════════════════════════════
let lastHighlightedGloss = null;
function renderGlossChips(glosses) {
    UI.glossBox.innerHTML = glosses.map(g =>
        `<span class="gloss-chip" id="gc-${g}" onclick="loadVideo('${g}')">${g}</span>`
    ).join('');
}
function highlightGloss(g) {
    if (g === lastHighlightedGloss) return;
    lastHighlightedGloss = g;
    document.querySelectorAll('.gloss-chip').forEach(el =>
        el.classList.toggle('active', el.innerText === g)
    );
    loadVideo(g);
}

// ═══ UTILS ════════════════════════════════════════════════════════════════════
function showError(msg) {
    UI.errorMsg.style.display = msg ? 'block' : 'none';
    UI.errorMsg.innerText = msg ? `⚠️ ${msg}` : '';
}

// ═══ BUTTON EVENTS ════════════════════════════════════════════════════════════
UI.btnTranslate.addEventListener('click', translate);
UI.textInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); translate(); } });
UI.btnPlay.addEventListener('click', () => {
    isPlaying = !isPlaying;
    UI.btnPlay.innerText = isPlaying ? '⏸ Pause' : '▶ Play';
});
UI.btnStop.addEventListener('click', () => {
    isPlaying = false;
    currentFrame = 0;
    lastHighlightedGloss = null;
    UI.btnPlay.innerText = '▶ Play';
    UI.frameInfo.innerText = 'Frame: 0 / —';
    ctx.clearRect(0, 0, skCanvas.width, skCanvas.height);
});
UI.modeToggle.addEventListener('click', () => {
    const modes = ['both', 'skeleton', 'video', 'avatar'];
    const labels = { both: '🔀 Todos', skeleton: '📊 Solo Esqueleto', video: '🎥 Solo Video', avatar: '🤖 Solo Avatar' };
    viewMode = modes[(modes.indexOf(viewMode) + 1) % modes.length];
    UI.modeToggle.innerText = labels[viewMode];
    document.getElementById('skeleton-panel').style.display = (['both', 'skeleton'].includes(viewMode)) ? 'flex' : 'none';
    document.getElementById('video-panel').style.display = (['both', 'video'].includes(viewMode)) ? 'flex' : 'none';
    document.getElementById('avatar-panel').style.display = (['both', 'avatar'].includes(viewMode)) ? 'block' : 'none';
    resizeCanvas();
    window.dispatchEvent(new Event('resize'));
});

// Set initial visibility
function setInitialView() {
    const panels = {
        'skeleton': ['skeleton-panel'],
        'video': ['video-panel'],
        'avatar': ['avatar-panel'],
        'both': ['skeleton-panel', 'video-panel']
    };

    ['skeleton-panel', 'video-panel', 'avatar-panel'].forEach(id => {
        document.getElementById(id).style.display = 'none';
    });

    (panels[viewMode] || ['skeleton-panel']).forEach(id => {
        const el = document.getElementById(id);
        el.style.display = (id === 'avatar-panel') ? 'block' : 'flex';
    });
}

// ═══ INIT ═════════════════════════════════════════════════════════════════════
initVRM();
loadVocab();
setInitialView();

UI.status.innerText = 'Sistema listo. Visualizando Esqueleto de Movimiento.';
