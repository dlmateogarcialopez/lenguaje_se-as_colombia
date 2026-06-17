# 03 — Retargeting: landmarks → avatar 3D

Este es el **corazón** del proyecto. Convertir puntos 2D/3D de MediaPipe en rotaciones de huesos de un esqueleto humanoide VRM.

Toda la lógica vive en `static/index.html`. Esta documentación referencia las funciones por nombre.

## Conceptos previos

### Avatar VRM

Formato estándar de avatar humanoide. Cada avatar tiene un **`humanoid`** con huesos nombrados según una convención fija: `hips`, `spine`, `chest`, `neck`, `head`, `leftUpperArm`, `leftLowerArm`, `leftHand`, `leftIndexProximal`, etc.

Cargado con `@pixiv/three-vrm`:

```js
const loader = new GLTFLoader();
loader.register(p => new VRMLoaderPlugin(p));
loader.load('avatar.vrm', gltf => {
  vrm = gltf.userData.vrm;
  if (vrm.meta?.metaVersion === '0') VRMUtils.rotateVRM0(vrm);  // VRM0 mira hacia +Z, VRM1 hacia -Z
  scene.add(vrm.scene);
});
```

Acceso a un hueso:

```js
const codoIzq = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
```

### Pose "rest" (T-pose)

El avatar en reposo tiene:
- Brazos extendidos a los lados (`leftUpperArm` apunta hacia +X, `rightUpperArm` hacia -X)
- Cuerpo erguido (`spine` apunta hacia +Y)
- Mirando hacia -Z (después de `rotateVRM0`)

**Cada hueso tiene una dirección de reposo** que se calcula al cargar el avatar:

```js
function calcularRest() {
  restDirs = {};
  const cadenas = [
    ['leftUpperArm', 'leftLowerArm'],   // dir = de hombro a codo
    ['leftLowerArm', 'leftHand'],       // dir = de codo a muñeca
    ['rightUpperArm', 'rightLowerArm'],
    ['rightLowerArm', 'rightHand'],
  ];
  for (const [hueso, hijo] of cadenas) {
    const nh = vrm.humanoid.getNormalizedBoneNode(hijo);
    if (nh) restDirs[hueso] = nh.position.clone().normalize();
  }
}
```

`nh.position` es la posición local del hijo respecto al padre — esto da la dirección natural del hueso en T-pose.

## Mapeo de ejes: imagen → mundo 3D

Las coordenadas de MediaPipe vienen del espacio de **imagen** (origen arriba-izquierda, Y crece hacia abajo). El avatar vive en espacio 3D con Y hacia arriba. Conversión:

```js
const Z_FACTOR = 0.65;  // z de MediaPipe es ruidoso, amortiguar
function dirLm(arr, a, b) {
  const [xs, ys, zs] = arr;
  return new THREE.Vector3(
    xs[b] - xs[a],          // X igual
    -(ys[b] - ys[a]),       // Y invertida
    -(zs[b] - zs[a]) * Z_FACTOR
  ).normalize();
}
```

Esta función devuelve la **dirección unitaria** del landmark `a` al landmark `b` en espacio del mundo del avatar.

## El truco central: `apuntarHueso()`

Dado un vector dirección objetivo (de los landmarks) y un hueso, calcula la rotación que hace que ese hueso apunte en esa dirección. Cuatro pasos:

```js
function apuntarHueso(nombre, targetDir, lerp = 0.55) {
  const hueso = vrm.humanoid.getNormalizedBoneNode(nombre);

  // 1. Obtener rotación del PADRE en mundo (porque el hueso rota en espacio local)
  hueso.parent.updateWorldMatrix(true, false);
  hueso.parent.getWorldQuaternion(_q1);

  // 2. Llevar targetDir (mundo) al espacio local del padre
  const local = targetDir.clone().applyQuaternion(_q2.copy(_q1).invert());

  // 3. Quaternion que rota desde la dirección de reposo a la dirección objetivo
  _q3.setFromUnitVectors(restDirs[nombre], local);

  // 4. Slerp para suavizar (no aplicar 100%, solo 55%)
  hueso.quaternion.slerp(_q3, lerp);
}
```

### ¿Por qué espacio local?

Three.js / VRM aplican rotaciones jerárquicamente: `leftLowerArm` está dentro de `leftUpperArm`. Si rotas el hombro 90°, el codo rota con él. Para que el codo apunte a una dirección absoluta, hay que **cancelar** la rotación heredada del hombro convirtiendo el target al espacio local del padre.

### ¿Por qué `setFromUnitVectors`?

Es el método más simple para "rotar A para que apunte a B": calcula el quaternion de eje perpendicular y ángulo entre ambos. Funciona porque el hueso siempre apunta a su hijo en T-pose.

## Aplicación por frame: `aplicarFrame(f)`

Orden de operaciones por cada frame del video:

### 1. Actualizar esqueleto visual (debug)

```js
actualizarEsqueleto(f);  // renderiza puntos azules/naranjas al lado del avatar
```

Esto es solo visual. No afecta al avatar.

### 2. Torso (línea de hombros)

```js
const sh = new THREE.Vector3(xs[11] - xs[12], -(ys[11] - ys[12]), 0).normalize();
_q3.setFromUnitVectors(_restX, sh);  // _restX = (1,0,0): línea de hombros en T-pose
chest.quaternion.slerp(clampQuat(_q3, 0.17), 0.15);
```

- Se usa solo X/Y (sin Z) → evita que ruido de profundidad gire al avatar.
- `clampQuat(_q3, 0.17)` recorta la rotación a máximo ~10° → evita giros bruscos.

### 3. Cabeza

```js
const dirCabeza = new THREE.Vector3(xs[0] - cx, -(ys[0] - cy), 0).normalize();
_q3.setFromUnitVectors((0,1,0), dirCabeza);  // cabeza en reposo apunta a +Y
neck.quaternion.slerp(clampQuat(_q3, 0.35), 0.2);
```

Donde `cx, cy` = centro entre hombros. Máximo ~20° de inclinación.

### 4. Brazos

```js
apuntarHueso('leftUpperArm',  dirLm(f.pose, 11, 13));  // hombro → codo
apuntarHueso('leftLowerArm',  dirLm(f.pose, 13, 15));  // codo → muñeca
apuntarHueso('rightUpperArm', dirLm(f.pose, 12, 14));
apuntarHueso('rightLowerArm', dirLm(f.pose, 14, 16));
```

Esto es lo más simple y robusto: cuatro vectores → cuatro rotaciones.

### 5. Asignación de manos

⚠️ **Problema crítico**: la etiqueta `Left/Right` de MediaPipe no es confiable.

Solución: asignar cada mano detectada a la muñeca corporal más cercana.

```js
function asignarManos(f) {
  const [xs, ys] = f.pose;
  const dist = (h, wi) => Math.hypot(h[0][0] - xs[wi], h[1][0] - ys[wi]);
  const manos = [f.l_hand, f.r_hand].filter(h => h[0].length === 21);

  if (manos.length === 1) {
    const h = manos[0];
    return dist(h, 15) <= dist(h, 16) ? { l: h, r: VACIA } : { l: VACIA, r: h };
  }
  // Asignación que minimiza distancia total
  const [a, b] = manos;
  return dist(a, 15) + dist(b, 16) <= dist(a, 16) + dist(b, 15)
    ? { l: a, r: b } : { l: b, r: a };
}
```

Los índices 15 y 16 son las muñecas en MediaPipe Pose. La distancia se mide entre la muñeca de la pose y el landmark `0` (muñeca) de cada mano detectada.

### 6. Orientación de muñeca: `orientarMuneca()`

La muñeca no apunta a un solo punto: tiene **3 ejes** (dirección de los dedos, normal a la palma, lateral). Se construye una base ortonormal:

```js
function orientarMuneca(nombre, hand, lado, lerp = 0.55) {
  const dir = dirLm(hand, 0, 9);                  // muñeca → nudillo medio
  const vI  = dirLm(hand, 0, 5);                  // muñeca → nudillo índice
  const vP  = dirLm(hand, 0, 17);                 // muñeca → nudillo meñique
  let normal = new THREE.Vector3().crossVectors(vI, vP).normalize();
  if (lado === 'right') normal.negate();          // palma apunta al lado correcto
  const t3 = new THREE.Vector3().crossVectors(dir, normal).normalize();
  const n2 = new THREE.Vector3().crossVectors(t3, dir).normalize();
  _m1.makeBasis(dir, n2, t3);                     // matriz objetivo

  // Base de reposo (T-pose)
  const rD = new THREE.Vector3(lado === 'left' ? 1 : -1, 0, 0);
  const rN = new THREE.Vector3(0, -1, 0);
  const r3 = new THREE.Vector3().crossVectors(rD, rN);
  _m2.makeBasis(rD, rN, r3);

  // Rotación = M_objetivo · M_reposo⁻¹
  _q3.setFromRotationMatrix(_m1.multiply(_m2.invert()));

  // Llevar a espacio local del padre
  hueso.parent.getWorldQuaternion(_q1);
  const localQ = _q2.copy(_q1).invert().multiply(_q3);
  hueso.quaternion.slerp(localQ, lerp);
}
```

### 7. Dedos (curls)

Para los dedos no se usa retargeting geométrico, sino la librería **Kalidokit** que ya tiene un solver dedicado:

```js
const rig = Kalidokit.Hand.solve(aLandmarks(mano), 'Left');
// rig.LeftIndexProximal = { x, y, z } en radianes
for (const d of ['Thumb','Index','Middle','Ring','Little']) {
  for (const ph of ['Proximal','Intermediate','Distal']) {
    rotarHueso('Left' + d + ph, rig['Left' + d + ph]);
  }
}
```

Kalidokit devuelve rotaciones en Euler. `rotarHueso()` las aplica:

```js
function rotarHueso(nombre, rot, amortiguar = 1, lerp = 0.45) {
  const hueso = vrm.humanoid.getNormalizedBoneNode(nombreHueso(nombre));
  _euler.set(rot.x * amortiguar, rot.y * amortiguar, rot.z * amortiguar, 'XYZ');
  _quat.setFromEuler(_euler);
  hueso.quaternion.slerp(_quat, lerp);
}
```

### Pulgar VRM1 vs VRM0

VRM1 renombró los huesos del pulgar. Kalidokit usa nombres VRM0:

```js
const NOMBRE_VRM1 = {
  LeftThumbProximal:     'leftThumbMetacarpal',
  LeftThumbIntermediate: 'leftThumbProximal',
  LeftThumbDistal:       'leftThumbDistal',
  // ... mismo para Right
};
```

## Suavizado temporal

Tres mecanismos previenen tirones:

1. **Interpolación entre frames** (`interpolar()`): mezcla linealmente landmarks de frame N y N+1 según `frameTime - i`.
2. **Slerp con factor `lerp` < 1**: cada hueso se mueve solo 45-55% hacia el objetivo por frame → inercia natural.
3. **`clampQuat()`** en torso y cabeza: limita el ángulo máximo de rotación.

## Bucle de animación

```js
function animar() {
  requestAnimationFrame(animar);
  const dt = reloj.getDelta();

  if (!actual && cola.length) {
    actual = cola.shift();
    // mostrar video sincronizado...
  }
  if (actual) {
    frameTime += dt * actual.fps;
    const i = Math.floor(frameTime);
    if (i + 1 < actual.frames.length) {
      aplicarFrame(interpolar(actual.frames[i], actual.frames[i+1], frameTime - i));
    } else if (i < actual.frames.length) {
      aplicarFrame(actual.frames[i]);
    } else {
      actual = null;  // seña terminada
    }
  } else {
    posturaReposo();  // slerp todos los huesos hacia identidad
  }

  vrm.update(dt);
  renderer.render(scene, camera);
}
```

`frameTime` avanza a la velocidad real del video (`fps * dt`). Esto permite reproducir a la misma velocidad sin importar el framerate del navegador.
