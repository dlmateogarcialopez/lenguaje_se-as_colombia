// visualizer_app.js — LSC Triple-Mode Visualizer v3.0
// Panel 1: Skeleton canvas  — MediaPipe landmarks in real time
// Panel 2: Real signer video — original LSC50 dataset
// Panel 3: Canvas avatar    — direct landmark rendering (guaranteed sync)

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

// View mode: 'both' | 'skeleton' | 'video' | 'avatar'
let viewMode = 'both';

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
    // Head
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    // Shoulders → hips
    [11, 12], [11, 23], [12, 24], [23, 24],
    // Left arm
    [11, 13], [13, 15],
    // Right arm
    [12, 14], [14, 16],
    // Left hand (from wrist)
    [15, 17], [15, 19], [15, 21], [17, 19],
    // Right hand (from wrist)
    [16, 18], [16, 20], [16, 22], [18, 20],
    // Left leg
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
    // Right leg
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// MediaPipe Hand connections (21 landmarks)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
    [0, 13], [13, 14], [14, 15], [15, 16],// Ring
    [0, 17], [17, 18], [18, 19], [19, 20],// Little
    [5, 9], [9, 13], [13, 17],          // Palm
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

// Use ResizeObserver for reliable sizing after layout
const skPanel = document.getElementById('skeleton-panel');
const ro = new ResizeObserver(() => resizeCanvas());
ro.observe(skPanel);

// Also call after first paint
requestAnimationFrame(() => { resizeCanvas(); });
window.addEventListener('resize', resizeCanvas);

// ═══ SKELETON DRAW ENGINE ═══════════════════════════════════════════════════════
function drawSkeleton(frame) {
    const W = skCanvas.width;
    const H = skCanvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background glow
    const radGrad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.min(W, H) * 0.5);
    radGrad.addColorStop(0, 'rgba(0,237,218,0.04)');
    radGrad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = radGrad;
    ctx.fillRect(0, 0, W, H);

    // ── BODY POSE ──────────────────────────────────────────────────────────────
    const pose = frame.poseLandmarks;
    if (pose && pose.length > 0) {
        drawPoseSkeleton(pose, W, H);
    }

    // ── FACE ───────────────────────────────────────────────────────────────────
    const face = frame.faceLandmarks;
    if (face && face.length > 0) {
        drawFace(face, W, H);
    }

    // ── HANDS ──────────────────────────────────────────────────────────────────
    const rh = frame.rightHandLandmarks;
    const lh = frame.leftHandLandmarks;
    if (rh && rh.length > 0) drawHand(rh, COLORS.rhand, W, H);
    if (lh && lh.length > 0) drawHand(lh, COLORS.lhand, W, H);

    // Gloss label
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
    // Map landmark x,y from [0,1] to canvas, with padding
    const x = (padX + lm.x * (1 - 2 * padX)) * W;
    const y = (padY + lm.y * (1 - 2 * padY)) * H;
    return { x, y };
}

function drawPoseSkeleton(landmarks, W, H) {
    // Connections
    POSE_CONNECTIONS.forEach(([a, b]) => {
        const lmA = landmarks[a], lmB = landmarks[b];
        if (!lmA || !lmB) return;
        const pA = mapLM(lmA, W, H);
        const pB = mapLM(lmB, W, H);

        // Color segments by body zone
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

    // Joints
    landmarks.forEach((lm, i) => {
        if (!lm) return;
        const p = mapLM(lm, W, H);
        const isKeyJoint = [0, 11, 12, 13, 14, 15, 16, 23, 24].includes(i);
        const r = isKeyJoint ? 7 : 4;

        let col = COLORS.body;
        if (i <= 10) col = COLORS.face;          // face
        if (i >= 23) col = COLORS.spine;          // legs

        // Glow effect
        ctx.shadowColor = col;
        ctx.shadowBlur = isKeyJoint ? 14 : 6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Labels for key joints
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
    // Determine offset: hands in mediapipe are 0-1 relative
    // We overlay them near where wrist landmark is
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
    // Draw a simplified face contour — just key face landmarks
    // MediaPipe Face: 468 landmarks, we draw a sampled set
    const keyIndices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        // eyes
        33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
        // mouth
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ];

    ctx.globalAlpha = 0.35;
    keyIndices.forEach(i => {
        const lm = landmarks[i];
        if (!lm) return;
        const p = mapLM(lm, W, H);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.face;
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
            // Draw skeleton
            if (viewMode !== 'avatar' && viewMode !== 'video') drawSkeleton(frame);
            // Draw canvas avatar (replaces VRM)
            if (viewMode !== 'skeleton' && viewMode !== 'video') drawAvatar(frame);
            // Update gloss UI
            if (frame.gloss) highlightGloss(frame.gloss);
            UI.frameInfo.innerText = `Frame: ${currentFrame + 1} / ${motionData.length}`;
        }
        currentFrame = (currentFrame + 1) % motionData.length;
        lastTime = time;
    }
}
requestAnimationFrame(animLoop);

// ═══ AVATAR CANVAS SETUP ══════════════════════════════════════════════════════
const avCanvas = document.getElementById('avatar-canvas');
const avCtx = avCanvas.getContext('2d');

const avPanel = document.getElementById('avatar-panel');
const avRO = new ResizeObserver(() => resizeAvatarCanvas());
avRO.observe(avPanel);

function resizeAvatarCanvas() {
    const w = avPanel.offsetWidth, h = avPanel.offsetHeight;
    if (w > 0 && h > 0) { avCanvas.width = w; avCanvas.height = h; }
}
requestAnimationFrame(() => resizeAvatarCanvas());

// ═══ AVATAR RENDERING ENGINE ══════════════════════════════════════════════════
// Draws a stylized human figure directly from MediaPipe landmarks.
// This guarantees EXACT sync with the skeleton panel and the real video.

const AV = {
    skin: '#d4956a',
    skinD: '#b57a50',
    shirt: '#8B5CF6',
    shirtD: '#6D28D9',
    pants: '#1e293b',
    pantsD: '#0f172a',
    rHand: '#ff9f8b',
    lHand: '#8bc4ff',
    face: '#e8b89a',
    faceD: '#c49278',
};

function avMap(lm, W, H, padX = 0.06, padY = 0.04) {
    const x = (padX + lm.x * (1 - 2 * padX)) * W;
    const y = (padY + lm.y * (1 - 2 * padY)) * H;
    return { x, y };
}

function drawThickLine(ctx, p1, p2, w, col1, col2) {
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const nx = -dy / len * w / 2, ny = dx / len * w / 2;
    const grad = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
    grad.addColorStop(0, col1);
    grad.addColorStop(1, col2);
    ctx.beginPath();
    ctx.moveTo(p1.x + nx, p1.y + ny);
    ctx.lineTo(p2.x + nx, p2.y + ny);
    ctx.lineTo(p2.x - nx, p2.y - ny);
    ctx.lineTo(p1.x - nx, p1.y - ny);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
}

function drawJoint(ctx, p, r, col) {
    ctx.shadowColor = col;
    ctx.shadowBlur = r * 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.shadowBlur = 0;
}

function drawFaceCircle(ctx, nose, lSh, rSh, W, H, faceLms) {
    const midShX = (lSh.x + rSh.x) / 2;
    const midShY = (lSh.y + rSh.y) / 2;
    const shoulderW = Math.abs(rSh.x - lSh.x);

    let headCX, headCY, headR;

    if (faceLms && faceLms.length >= 468) {
        // Fit circle exactly to the detected Face Mesh
        const fTop = avMap(faceLms[10], W, H);     // Top edge of face
        const fBottom = avMap(faceLms[152], W, H); // Chin
        const fLeft = avMap(faceLms[234], W, H);   // Left cheek edge
        const fRight = avMap(faceLms[454], W, H);  // Right cheek edge

        headCX = (fLeft.x + fRight.x) / 2;
        headCY = (fTop.y + fBottom.y) / 2;

        // Radius based on face width/height to fully encompass the features
        headR = Math.max(20, (fBottom.y - fTop.y) * 0.65, (fRight.x - fLeft.x) * 0.65);
    } else {
        // Fallback: estimate head strictly from body landmarks
        headCX = nose.x * W * (1 - 2 * 0.06) + W * 0.06;
        headCY = midShY * H * (1 - 2 * 0.04) + H * 0.04 - shoulderW * H * 0.8;
        headR = Math.max(20, shoulderW * W * 0.32);
    }

    // Head base shadow
    ctx.shadowColor = AV.faceD;
    ctx.shadowBlur = 8;
    const gHead = ctx.createRadialGradient(headCX - headR * 0.15, headCY - headR * 0.15, headR * 0.1, headCX, headCY, headR);
    gHead.addColorStop(0, AV.face);
    gHead.addColorStop(1, AV.faceD);
    ctx.beginPath();
    ctx.arc(headCX, headCY, headR, 0, Math.PI * 2);
    ctx.fillStyle = gHead;
    ctx.fill();
    ctx.shadowBlur = 0;

    if (faceLms && faceLms.length >= 468) {
        // Draw facial features exactly from landmarks
        const mapLM = (idx) => avMap(faceLms[idx], W, H);

        // Draw Eyes (simplified paths)
        const drawEye = (pts) => {
            ctx.beginPath();
            ctx.moveTo(mapLM(pts[0]).x, mapLM(pts[0]).y);
            for (let i = 1; i < pts.length; i++) ctx.lineTo(mapLM(pts[i]).x, mapLM(pts[i]).y);
            ctx.closePath();
            ctx.fillStyle = '#1a1a2e';
            ctx.fill();
        };
        // Left Eye: 33, 160, 158, 133, 153, 144
        drawEye([33, 160, 158, 133, 153, 144]);
        // Right Eye: 263, 387, 385, 362, 380, 373
        drawEye([263, 387, 385, 362, 380, 373]);

        // Draw Eyebrows (thick lines)
        const drawBrow = (pts) => {
            ctx.beginPath();
            ctx.moveTo(mapLM(pts[0]).x, mapLM(pts[0]).y);
            for (let i = 1; i < pts.length; i++) ctx.lineTo(mapLM(pts[i]).x, mapLM(pts[i]).y);
            ctx.strokeStyle = '#4a3b32';
            ctx.lineWidth = 2.5;
            ctx.stroke();
        };
        drawBrow([46, 53, 52, 65, 55]); // Left Brow
        drawBrow([276, 283, 282, 295, 285]); // Right Brow

        // Draw Mouth
        const drawLip = (pts, color, fill = false) => {
            ctx.beginPath();
            ctx.moveTo(mapLM(pts[0]).x, mapLM(pts[0]).y);
            for (let i = 1; i < pts.length; i++) ctx.lineTo(mapLM(pts[i]).x, mapLM(pts[i]).y);
            ctx.closePath();
            if (fill) {
                ctx.fillStyle = color;
                ctx.fill();
            } else {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        };

        // Inner mouth opening (background)
        drawLip([78, 81, 13, 311, 308, 317, 14, 87], '#301010', true);
        // Outer lip outline
        drawLip([61, 39, 0, 269, 291, 375, 17, 146], '#8b5e4a');

    } else {
        // Fallback static face
        const eyeY = headCY - headR * 0.15;
        const eyeOffX = headR * 0.3;
        [[headCX - eyeOffX, eyeY], [headCX + eyeOffX, eyeY]].forEach(([ex, ey]) => {
            ctx.beginPath();
            ctx.ellipse(ex, ey, headR * 0.12, headR * 0.08, 0, 0, Math.PI * 2);
            ctx.fillStyle = '#1a1a2e';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(ex + headR * 0.04, ey - headR * 0.02, headR * 0.03, 0, Math.PI * 2);
            ctx.fillStyle = 'white';
            ctx.fill();
        });

        // Mouth
        ctx.beginPath();
        ctx.arc(headCX, headCY + headR * 0.3, headR * 0.2, 0.1 * Math.PI, 0.9 * Math.PI);
        ctx.strokeStyle = '#8b5e4a';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    return { cx: headCX, cy: headCY, r: headR };
}

function drawTorso(ctx, lSh, rSh, lHip, rHip, W, H) {
    const pLS = avMap(lSh, W, H), pRS = avMap(rSh, W, H);
    const pLH = avMap(lHip, W, H), pRH = avMap(rHip, W, H);
    // Torso polygon
    const grad = ctx.createLinearGradient(pLS.x, pLS.y, pLH.x, pLH.y);
    grad.addColorStop(0, AV.shirt);
    grad.addColorStop(1, AV.shirtD);
    ctx.beginPath();
    ctx.moveTo(pLS.x, pLS.y);
    ctx.lineTo(pRS.x, pRS.y);
    ctx.lineTo(pRH.x, pRH.y);
    ctx.lineTo(pLH.x, pLH.y);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
}

function drawArm(ctx, sh, el, wr, W, H, color, colorD) {
    const pSh = avMap(sh, W, H), pEl = avMap(el, W, H), pWr = avMap(wr, W, H);
    const upperW = 14, lowerW = 10;
    drawThickLine(ctx, pSh, pEl, upperW, color, colorD);
    drawThickLine(ctx, pEl, pWr, lowerW, colorD, AV.skin);
    drawJoint(ctx, pEl, 7, colorD);
    drawJoint(ctx, pWr, 5, AV.skin);
}

function drawLeg(ctx, hip, knee, ankle, W, H) {
    const pH = avMap(hip, W, H), pK = avMap(knee, W, H), pA = avMap(ankle, W, H);
    drawThickLine(ctx, pH, pK, 16, AV.pants, AV.pantsD);
    drawThickLine(ctx, pK, pA, 13, AV.pantsD, '#334155');
    drawJoint(ctx, pK, 7, AV.pantsD);
    drawJoint(ctx, pA, 5, '#475569');
}

function drawHandLandmarks(ctx, landmarks, color, W, H) {
    if (!landmarks || landmarks.length < 21) return;
    const FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
    FINGERS.forEach(finger => {
        ctx.beginPath();
        for (let i = 0; i < finger.length; i++) {
            const lm = landmarks[finger[i]];
            if (!lm) continue;
            const p = avMap(lm, W, H);
            i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.globalAlpha = 0.9;
        ctx.stroke();
        ctx.globalAlpha = 1;
    });
    landmarks.forEach(lm => {
        if (!lm) return;
        const p = avMap(lm, W, H);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
    });
}

function drawAvatar(frame) {
    const W = avCanvas.width, H = avCanvas.height;
    if (!W || !H) return;
    avCtx.clearRect(0, 0, W, H);

    // Background gradient
    const bg = avCtx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.min(W, H) * 0.55);
    bg.addColorStop(0, 'rgba(123,94,167,0.06)');
    bg.addColorStop(1, 'rgba(0,0,0,0)');
    avCtx.fillStyle = bg;
    avCtx.fillRect(0, 0, W, H);

    const pose = frame.poseLandmarks;
    if (!pose || pose.length < 29) return;

    const P = i => pose[i];
    const lSh = P(11), rSh = P(12), lEl = P(13), rEl = P(14);
    const lWr = P(15), rWr = P(16), lHip = P(23), rHip = P(24);
    const lKn = P(25), rKn = P(26), lAn = P(27), rAn = P(28);
    const nose = P(0);

    // Draw order: back → front (legs, torso, arms, head, hands)
    if (lHip && rHip && lKn && lAn) drawLeg(avCtx, lHip, lKn, lAn, W, H);
    if (rHip && rKn && rAn) drawLeg(avCtx, rHip, rKn, rAn, W, H);

    if (lSh && rSh && lHip && rHip) drawTorso(avCtx, lSh, rSh, lHip, rHip, W, H);

    // Left arm (blue tones)
    if (lSh && lEl && lWr) drawArm(avCtx, lSh, lEl, lWr, W, H, '#6bcbff', '#3b82f6');
    // Right arm (purple tones)
    if (rSh && rEl && rWr) drawArm(avCtx, rSh, rEl, rWr, W, H, AV.shirt, AV.shirtD);

    // Shoulder dots
    if (lSh) drawJoint(avCtx, avMap(lSh, W, H), 9, '#6bcbff');
    if (rSh) drawJoint(avCtx, avMap(rSh, W, H), 9, AV.shirt);

    // Head
    if (nose && lSh && rSh) {
        drawFaceCircle(avCtx, nose, lSh, rSh, W, H, frame.faceLandmarks);
    }

    // Hands from hand landmarks
    drawHandLandmarks(avCtx, frame.rightHandLandmarks, '#ff9f8b', W, H);
    drawHandLandmarks(avCtx, frame.leftHandLandmarks, '#8bc4ff', W, H);

    // Gloss label
    if (frame.gloss) {
        avCtx.font = 'bold 22px Outfit, sans-serif';
        avCtx.fillStyle = '#7B5EA7';
        avCtx.textAlign = 'center';
        avCtx.globalAlpha = 0.9;
        avCtx.fillText(frame.gloss, W / 2, H - 24);
        avCtx.globalAlpha = 1;
    }
}

// ═══ STUB: no VRM to initialize ════════════════════════════════════════════════
function initVRM() {
    // Canvas avatar is ready immediately
    const vrmLoadEl = document.getElementById('vrm-loading');
    if (vrmLoadEl) vrmLoadEl.style.display = 'none';
    UI.status.innerText = 'Avatar listo. Escribe texto para traducir señas.';
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

    // Show loading state
    vidPlch.style.display = 'none';
    vidEl.style.display = 'none';
    vidError.style.display = 'none';
    vidLoad.style.display = 'flex';
    vidBadge.style.display = 'none';

    document.getElementById('video-panel-label').innerText = `🎥 Video Real — ${gloss}`;

    try {
        // Pre-fetch to check if the server can serve it
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
    // Load video for this word immediately when clicked
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
});

// ═══ INIT ═════════════════════════════════════════════════════════════════════
initVRM();
loadVocab();

UI.status.innerText = 'Sistema listo. Cargando avatar...';
