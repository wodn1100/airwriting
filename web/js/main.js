/**
 * main.js — AirWriting 3D 디지털 트윈 (v2: 흰색 테마 + 모핑 + 학습)
 *
 * 변경사항:
 * - 흰색 배경 3D 씬
 * - 모핑 효과: 필기 궤적 → 학습 데이터 부드러운 전환
 * - 학습 UI: Record/Train 반자동 파이프라인
 * - 전완(forearm) 자유도: 원점 고정 해제 → 회전+이동 추적
 */

// ── Scene ──
const container = document.getElementById("canvas-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8fafc); // 화이트보드 배경

// 1인칭(정면) 시점: 극학의 망원(Telephoto) 세팅으로 완전한 평면 직교(Orthographic) 느낌 구현
// 카메라를 아주 멀리(Z=3) 배치하고 FOV를 극도로 좁혀(15도) 원근 왜곡을 0으로 만듭니다.
const camera = new THREE.PerspectiveCamera(15, window.innerWidth / window.innerHeight, 0.1, 50);
// 사용자 손(Y=-0.25)의 중심을 바라보도록 카메라 위치를 한참 뒤로 뺍니다.
camera.position.set(0, -0.05, 3.5); 

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
// 시야 센터를 팔이 뻗어지는 중심(Y=-0.25, 칠판쪽)으로 고정
controls.target.set(0, -0.05, -0.6); 

// 사용자가 화면을 고정해서 보도록 회전 제한
controls.maxAzimuthAngle = Math.PI / 16;
controls.minAzimuthAngle = -Math.PI / 16;
controls.maxPolarAngle = Math.PI / 2 + Math.PI / 16;
controls.minPolarAngle = Math.PI / 2 - Math.PI / 16;
controls.enableRotate = false; // 흔들림 방지를 위해 1인칭 뷰 고정
controls.enablePan = false; 
controls.maxDistance = 6.0;
controls.minDistance = 0.1;

// ── Lighting ──
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.5);
keyLight.position.set(0, 1, 2);
scene.add(keyLight);

// ── Grid & Whiteboard (기본 숨김 - 칠판 모드) ──
const groundGeom = new THREE.PlaneGeometry(4, 4);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0xf8fafc, roughness: 1.0, metalness: 0, transparent: true, opacity: 0.5,
  side: THREE.DoubleSide
});
const ground = new THREE.Mesh(groundGeom, groundMat);
// 카메라(z=0.1) → 글씨 궤적(z=-0.4) → 화이트보드(z=-0.7)
ground.position.z = -0.7;
scene.add(ground);

const gridHelper = new THREE.GridHelper(4, 40, 0x94a3b8, 0xcbd5e1);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.6;
// 그리드도 세로(칠판) 방향으로 세워줍니다.
gridHelper.rotation.x = Math.PI / 2;
gridHelper.position.z = -0.69; 
scene.add(gridHelper);
const axesHelper = new THREE.AxesHelper(0.1);
// AxesHelper도 세워진 평면의 중심에 맞춥니다.
axesHelper.position.z = -0.68;
scene.add(axesHelper);

// 그리드가 켜진 상태를 디폴트로 설정
ground.visible = true;
gridHelper.visible = true;
axesHelper.visible = true;

// ── Arm Model ──
const COLORS = {
  forearm: 0x4f6ef7,
  hand: 0x10b981,
  finger: 0xa78bfa,
  tip: 0xef4444, // 포인터(팁)는 빨간 점으로 유지
};

function makeJoint(color, size = 0.012) {
  const mat = new THREE.MeshPhysicalMaterial({
    color, metalness: 0.15, roughness: 0.35, clearcoat: 0.6,
    emissive: color, emissiveIntensity: 0.08,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 16, 16), mat);
  mesh.castShadow = true;
  scene.add(mesh);
  return mesh;
}

const joints = {
  forearm: makeJoint(COLORS.forearm, 0.014),
  hand: makeJoint(COLORS.hand),
  finger: makeJoint(COLORS.finger),
};

// ── 1. 입체 분필(펜) 모델 생성 ──
const penGeom = new THREE.CylinderGeometry(0.003, 0.008, 0.1, 16); // 끝이 뾰족한 펜 모양 (길이 10cm)
penGeom.rotateX(Math.PI / 2); // 기본 실린더 길이(Y축)를 앞뒤(Z축) 방향으로 눕힘
// 펜의 중심점(Origin)이 '펜촉 앞부분'에 오도록 바디 플랫을 뒤쪽(+Z방향)으로 밀어줍니다.
// 이렇게 해야 센서에서 넘어온 정확한 fingertip 좌표계와 화면상 펜촉의 위치, 그리고 그림자가 100% 일치합니다.
penGeom.translate(0, 0, 0.05); 

const penMat = new THREE.MeshPhysicalMaterial({ color: 0x333333, metalness: 0.2, roughness: 0.9 });
const fingertipMesh = new THREE.Mesh(penGeom, penMat);
scene.add(fingertipMesh);

// ── 2. 보드 투영 그림자(Shadow/Crosshair) 생성 ──
const projectionGeom = new THREE.RingGeometry(0.003, 0.012, 24);
const projectionMat = new THREE.MeshBasicMaterial({ 
    color: 0x000000, transparent: true, opacity: 0.2, side: THREE.DoubleSide 
});
const projectionMesh = new THREE.Mesh(projectionGeom, projectionMat);
projectionMesh.position.z = -0.68; // 칠판(Z=-0.69) 바로 앞
scene.add(projectionMesh);

// 팔 관절 숨기기 (깔끔한 UI 요청사항 도입)
joints.forearm.visible = false;
joints.hand.visible = false;
joints.finger.visible = false;

function makeBone(color) {
  const geom = new THREE.CylinderGeometry(0.003, 0.003, 1, 8);
  geom.translate(0, 0.5, 0);
  const mat = new THREE.MeshPhysicalMaterial({
    color, metalness: 0.1, roughness: 0.5, transparent: true, opacity: 0.6,
  });
  const mesh = new THREE.Mesh(geom, mat);
  scene.add(mesh);
  return mesh;
}

const bones = {
  forearm_hand: makeBone(COLORS.forearm),
  hand_finger: makeBone(COLORS.hand),
  finger_tip: makeBone(COLORS.finger),
};

// 뼈대 선 숨기기
bones.forearm_hand.visible = false;
bones.hand_finger.visible = false;
bones.finger_tip.visible = false;

function updateBone(boneMesh, from, to) {
  const dir = new THREE.Vector3().subVectors(to, from);
  const len = dir.length();
  if (len < 0.001) return;
  boneMesh.position.copy(from);
  boneMesh.scale.set(1, len, 1);
  boneMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
}

// ── Trajectory ──
const MAX_PTS = 8000;
const livePositions = new Float32Array(MAX_PTS * 3);
let liveCount = 0;
const liveGeom = new THREE.BufferGeometry();
liveGeom.setAttribute("position", new THREE.BufferAttribute(livePositions, 3));
liveGeom.setDrawRange(0, 0);

const liveMat = new THREE.LineBasicMaterial({ color: 0x0f172a, linewidth: 8, transparent: true, opacity: 0.95 });
const liveLine = new THREE.Line(liveGeom, liveMat);
scene.add(liveLine);

// ── Thick 3D Line using InstancedMesh ──
const liveCylGeom = new THREE.CylinderGeometry(0.007, 0.007, 1, 8); // radius 7mm
liveCylGeom.rotateX(Math.PI / 2); // Align to Z axis for setFromUnitVectors
const liveCylMat = new THREE.MeshBasicMaterial({ color: 0x0f172a, transparent: true, opacity: 0.95 });
const liveCylMesh = new THREE.InstancedMesh(liveCylGeom, liveCylMat, MAX_PTS);
liveCylMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
liveCylMesh.count = 0;
scene.add(liveCylMesh);

// ── Morphing target trajectory (학습 데이터) ──
let morphTargetPositions = null; // Float32Array
let morphProgress = 0;
let morphActive = false;
const MORPH_DURATION = 1500; // ms
let morphStartTime = 0;

const morphGeom = new THREE.BufferGeometry();
const morphPositions = new Float32Array(MAX_PTS * 3);
morphGeom.setAttribute("position", new THREE.BufferAttribute(morphPositions, 3));
morphGeom.setDrawRange(0, 0);
const morphMat = new THREE.LineBasicMaterial({
  color: 0x4f6ef7, linewidth: 2, transparent: true, opacity: 0,
});
const morphLine = new THREE.Line(morphGeom, morphMat);
scene.add(morphLine);

// particle glow (모핑 시 파티클 흩어지는 효과)
const morphParticleGeom = new THREE.BufferGeometry();
const morphParticlePositions = new Float32Array(200 * 3);
morphParticleGeom.setAttribute("position", new THREE.BufferAttribute(morphParticlePositions, 3));
const morphParticleMat = new THREE.PointsMaterial({
  color: COLORS.finger, size: 0.005, transparent: true, opacity: 0,
  blending: THREE.AdditiveBlending,
});
const morphParticles = new THREE.Points(morphParticleGeom, morphParticleMat);
scene.add(morphParticles);

function startMorph(templatePoints) {
  if (!templatePoints || templatePoints.length < 6) return;
  morphTargetPositions = templatePoints;
  morphActive = true;
  morphStartTime = performance.now();
  morphProgress = 0;
  document.getElementById("morph-indicator").classList.add("active");

  // "pop" animation on recognized char
  const charEl = document.getElementById("recognized-char");
  charEl.classList.add("pop");
  setTimeout(() => charEl.classList.remove("pop"), 400);
}

function updateMorph() {
  if (!morphActive || !morphTargetPositions) return;

  const elapsed = performance.now() - morphStartTime;
  morphProgress = Math.min(elapsed / MORPH_DURATION, 1.0);
  const t = easeInOutCubic(morphProgress);

  const srcCount = liveCount;
  const tgtCount = Math.floor(morphTargetPositions.length / 3);
  const count = Math.max(srcCount, tgtCount);

  // 보간: live → target
  for (let i = 0; i < count; i++) {
    const si = Math.min(i, srcCount - 1);
    const ti = Math.min(i, tgtCount - 1);

    const sx = si >= 0 ? livePositions[si * 3] : 0;
    const sy = si >= 0 ? livePositions[si * 3 + 1] : 0;
    const sz = si >= 0 ? livePositions[si * 3 + 2] : 0;

    const tx = morphTargetPositions[ti * 3];
    const ty = morphTargetPositions[ti * 3 + 1];
    const tz = morphTargetPositions[ti * 3 + 2];

    morphPositions[i * 3] = sx + (tx - sx) * t;
    morphPositions[i * 3 + 1] = sy + (ty - sy) * t;
    morphPositions[i * 3 + 2] = sz + (tz - sz) * t;
  }

  morphGeom.setDrawRange(0, count);
  morphGeom.attributes.position.needsUpdate = true;

  // 색상/투명도 전환
  liveMat.opacity = 0.85 * (1 - t);
  liveCylMat.opacity = 0.85 * (1 - t);
  morphMat.opacity = t * 0.9;

  // 파티클 흩어짐
  if (t > 0.1 && t < 0.9) {
    morphParticleMat.opacity = (1 - Math.abs(t - 0.5) * 2) * 0.6;
    for (let i = 0; i < 200; i++) {
      const pi = Math.floor(Math.random() * count);
      morphParticlePositions[i * 3] = morphPositions[pi * 3] + (Math.random() - 0.5) * 0.02 * (1 - t);
      morphParticlePositions[i * 3 + 1] = morphPositions[pi * 3 + 1] + (Math.random() - 0.5) * 0.02 * (1 - t);
      morphParticlePositions[i * 3 + 2] = morphPositions[pi * 3 + 2] + (Math.random() - 0.5) * 0.02 * (1 - t);
    }
    morphParticleGeom.attributes.position.needsUpdate = true;
  } else {
    morphParticleMat.opacity = 0;
  }

  if (morphProgress >= 1.0) {
    morphActive = false;
    document.getElementById("morph-indicator").classList.remove("active");
    liveMat.opacity = 0.85;
    morphMat.opacity = 0;
    morphParticleMat.opacity = 0;

    // live를 target으로 교체
    for (let i = 0; i < count * 3; i++) {
      livePositions[i] = morphTargetPositions[i] || 0;
    }
    liveCount = count;
    liveGeom.setDrawRange(0, liveCount);
    liveGeom.attributes.position.needsUpdate = true;
    
    // Rebuild InstancedMesh cylinders for the new morphed skeleton
    for(let j = 1; j < liveCount; j++) {
       const px = livePositions[(j-1)*3], py = livePositions[(j-1)*3+1], pz = livePositions[(j-1)*3+2];
       const cx = livePositions[j*3], cy = livePositions[j*3+1], cz = livePositions[j*3+2];
       const prev = new THREE.Vector3(px, py, pz);
       const curr = new THREE.Vector3(cx, cy, cz);
       const dir = new THREE.Vector3().subVectors(curr, prev);
       const len = dir.length();
       if (len > 0.0001 && !isNaN(len)) {
         const mid = new THREE.Vector3().addVectors(prev, curr).multiplyScalar(0.5);
         const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.normalize());
         liveCylMesh.setMatrixAt(j-1, new THREE.Matrix4().compose(mid, quat, new THREE.Vector3(1, 1, len)));
       } else {
         liveCylMesh.setMatrixAt(j-1, new THREE.Matrix4().makeScale(0,0,0));
       }
    }
    liveCylMesh.instanceMatrix.needsUpdate = true;
    liveCylMesh.count = Math.max(0, liveCount - 1);
    
    liveMat.opacity = 0.85;
    liveCylMat.opacity = 0.85;
  }
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function addLivePoint(x, y, z) {
  if (liveCount >= MAX_PTS) return;
  const i = liveCount * 3;
  livePositions[i] = x;
  livePositions[i + 1] = y;
  livePositions[i + 2] = z;

  if (liveCount > 0) {
    const px = livePositions[i - 3], py = livePositions[i - 2], pz = livePositions[i - 1];
    const prev = new THREE.Vector3(px, py, pz);
    const curr = new THREE.Vector3(x, y, z);
    const dir = new THREE.Vector3().subVectors(curr, prev);
    const len = dir.length();
    
    if (len > 0.0001 && !isNaN(len)) {
      const mid = new THREE.Vector3().addVectors(prev, curr).multiplyScalar(0.5);
      const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.normalize());
      const matrix = new THREE.Matrix4().compose(mid, quat, new THREE.Vector3(1, 1, len));
      liveCylMesh.setMatrixAt(liveCount - 1, matrix);
    } else {
      liveCylMesh.setMatrixAt(liveCount - 1, new THREE.Matrix4().makeScale(0, 0, 0));
    }
    liveCylMesh.instanceMatrix.needsUpdate = true;
  }

  liveCount++;
  liveCylMesh.count = Math.max(0, liveCount - 1);
  liveGeom.setDrawRange(0, liveCount);
  liveGeom.attributes.position.needsUpdate = true;
}

// ── WebSocket ──
let ws = null, wsConnected = false;
let frameCounter = 0, fpsCounter = 0, lastFpsTime = performance.now();

function connectWebSocket() {
  const url = `ws://${location.hostname || "localhost"}:8765`;
  ws = new WebSocket(url);
  ws.onopen = () => { wsConnected = true; updateStatus(true); };
  ws.onmessage = (e) => { 
    try { 
      const data = JSON.parse(e.data);
      if (data.type) {
        handleServerMessage(data);
      } else {
        handleFrame(data); 
      }
    } catch (err) {} 
  };
  ws.onclose = () => { wsConnected = false; updateStatus(false); setTimeout(connectWebSocket, 3000); };
  ws.onerror = () => {};
}

function updateStatus(connected) {
  document.getElementById("ws-status-dot").classList.toggle("connected", connected);
  document.getElementById("ws-status-text").textContent = connected ? "Connected" : "Disconnected";
}

function sendToServer(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

// handleServerMessage는 L989~L1020에서 완전하게 정의됩니다.

// ── Frame handler ──
let isRecording = false;
let recordedLabel = "";
let wasWriting = false;  // 획 분리용 상태 추적

// 영점 조절 오프셋 및 민감도(마우스 감도 역할)
let positionOffset = [0, 0, 0];
let rawFingertip = [0, 0, 0];
const SENSITIVITY = 0.30; // 커서가 움직이는 배율 (기존 0.5에서 극단적으로 더 줄임)

// ── 센서 장착 각도 보정 ──
// ICM20948이 대각선으로 장착된 경우, 이 값을 조정하세요
// 양수 = 시계방향 회전, 음수 = 반시계방향 회전 (도 단위)
let ROTATION_DEG = 0;  // DGKVF 방식에서는 자동으로 모델링되므로 0도 유지
let rotCos = Math.cos(ROTATION_DEG * Math.PI / 180);
let rotSin = Math.sin(ROTATION_DEG * Math.PI / 180);

function setRotationCorrection(deg) {
  ROTATION_DEG = deg;
  rotCos = Math.cos(deg * Math.PI / 180);
  rotSin = Math.sin(deg * Math.PI / 180);
  console.log(`Rotation correction set to ${deg}°`);
}

// 2D 회전 적용 (센서 좌표 → 보정 좌표)
function applyRotation(x, y) {
  return [
    x * rotCos - y * rotSin,
    x * rotSin + y * rotCos,
  ];
}

// ── 순간이동 방지 + EMA 스무딩 ──
let prevSmoothedTip = null;
let framesSinceStart = 0;
const LERP_FACTOR = 0.35;
const TELEPORT_THRESHOLD = 0.15;
const AUTO_CENTER_FRAME = 10;
const EMA_ALPHA = 0.25;  // EMA 스무딩 (0=느림, 1=즉시). 흔들림 줄이려면 낮추세요
let emaTip = null;

function lerpVec3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function vec3Dist(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

function recenterView() {
  // ray_hit 기반: rawFingertip[0] = ray_x, rawFingertip[2] = ray_z
  positionOffset[0] = rawFingertip[0];
  positionOffset[2] = rawFingertip[2];
  
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "reset_yaw" }));
  }
  
  prevSmoothedTip = null;
  emaTip = null;
  clearTrajectory();
}

function handleFrame(data) {
  frameCounter++;
  fpsCounter++;
  framesSinceStart++;

  // ── 좌표 매핑: 레이저 포인터 (Ray Casting) ──
  if (data.ray_hit) {
    const rx = data.ray_hit[0];
    const rz = data.ray_hit[1];
    rawFingertip[0] = rx;
    rawFingertip[2] = rz;

    if (framesSinceStart === AUTO_CENTER_FRAME) {
      recenterView();
    }

    // 디버그: 50프레임마다 원본 좌표 출력
    if (frameCounter % 50 === 0) {
      console.log(`ray_hit[0]=${rx.toFixed(4)} ray_hit[1]=${rz.toFixed(4)}`);
    }

    const ox = positionOffset[0], oz = positionOffset[2];

    // 센서 방향 → 화면 좌표 (반전/회전 보정 포함)
    const rotated = applyRotation(
      -(rx - ox) * SENSITIVITY,   // X축 반전
      (rz - oz) * SENSITIVITY     // +Z = 위 방향 (반전 불필요)
    );
    let t = [rotated[0], rotated[1], -0.4];

    // ── 순간이동 방지 ──
    if (prevSmoothedTip !== null) {
      const dist = vec3Dist(t, prevSmoothedTip);
      if (dist > TELEPORT_THRESHOLD) {
        t = lerpVec3(prevSmoothedTip, t, LERP_FACTOR);
      }
    }
    prevSmoothedTip = [t[0], t[1], t[2]];

    // ── EMA 스무딩 ──
    if (emaTip === null) {
      emaTip = [t[0], t[1], t[2]];
    } else {
      emaTip[0] = EMA_ALPHA * t[0] + (1 - EMA_ALPHA) * emaTip[0];
      emaTip[1] = EMA_ALPHA * t[1] + (1 - EMA_ALPHA) * emaTip[1];
      emaTip[2] = t[2];
    }

    fingertipMesh.position.set(emaTip[0], emaTip[1], emaTip[2]);

    projectionMesh.position.x = emaTip[0];
    projectionMesh.position.y = emaTip[1];

    if (data.fingertip) {
      data.fingertip[0] = emaTip[0];
      data.fingertip[1] = emaTip[1];
      data.fingertip[2] = -0.68;
    }
  }

  // ── 관절 위치 (스켈레톤 시각화용, 기존 유지) ──
  if (data.positions) {
    const ox = positionOffset[0], oz = positionOffset[2];
    for (const [name, pos] of Object.entries(data.positions)) {
      if (joints[name]) {
        const raw = applyRotation(
          -(pos[0] - ox) * SENSITIVITY,
          -(pos[2] - oz) * SENSITIVITY
        );
        joints[name].position.set(raw[0], raw[1], -0.4);
      }
    }
  }

  if (data.orientations) {
    for (const [name, q] of Object.entries(data.orientations)) {
      if (joints[name]) joints[name].quaternion.set(q[1], q[2], q[3], q[0]);
    }
    // 펜(분필)을 실제 손가락 센서 각도로 물리적 방향 동기화
    if (data.orientations.finger) {
      const fq = data.orientations.finger;
      fingertipMesh.quaternion.set(fq[1], fq[2], fq[3], fq[0]);
    }
  }

  // 1인칭(2D) 뷰일 때는 시야를 가리는 허공의 펜을 숨김
  fingertipMesh.visible = !isFirstPersonView;

  // Bones
  updateBone(bones.forearm_hand, joints.forearm.position, joints.hand.position);
  updateBone(bones.hand_finger, joints.hand.position, joints.finger.position);
  updateBone(bones.finger_tip, joints.finger.position, fingertipMesh.position);

  // Trajectory — 획 분리 (NaN으로 선 끊기)
  const currentlyWriting = data.is_writing && data.fingertip;
  if (currentlyWriting) {
    if (!wasWriting) {
      // 새 획 시작 → 이전 획과 분리
      addLivePoint(NaN, NaN, NaN);
    }
    addLivePoint(data.fingertip[0], data.fingertip[1], data.fingertip[2]);
    
    // 쓰는 중일 때 포인트 프로젝션 강조
    projectionMat.opacity = 0.8;
    projectionMat.color.setHex(0xef4444);
  } else {
    // 평상시는 연한 검은 그림자
    projectionMat.opacity = 0.15;
    projectionMat.color.setHex(0x000000); 
  }
  wasWriting = currentlyWriting;

  // Writing indicator
  document.getElementById("writing-indicator").classList.toggle("active", data.is_writing || false);

  // Recognition & morph
  if (data.recognition && data.recognition.class && data.recognition.above_threshold) {
    updateRecognition(data.recognition);
    if (data.recognition.template_trajectory) {
      startMorph(new Float32Array(data.recognition.template_trajectory));
    }
  }

  // Sensor HUD
  updateSensorHUD(data);
  document.getElementById("frame-count").textContent = frameCounter;

  // Edge detection for hardware button
  const buttonPressed = data.is_writing || false;
  const buttonJustPressed = buttonPressed && !window._prevWritingState;
  
  // Dynamic Auto-Recording
  if (isAutoRecordingEnabled) {
      if (autoRecordArmed && buttonJustPressed) {
          // First touch -> start recording
          autoRecordArmed = false;
          startDynamicRecording();
      } else if (isRecording) {
          if (buttonPressed) {
              // Continuing stroke -> cancel inactivity timeout
              if (recordTimeout) {
                  clearTimeout(recordTimeout);
                  recordTimeout = null;
              }
              document.getElementById("train-status").textContent = "🔴 Recording Stroke...";
          } else if (window._prevWritingState) {
              // Just released button -> Wait 1.0s before ending the sample
              document.getElementById("train-status").textContent = "⏳ Waiting for next stroke (1.0s)...";
              recordTimeout = setTimeout(() => {
                  stopRecordingAndRearm();
              }, 1000);
          }
      }
  }
  window._prevWritingState = buttonPressed;

  // If recording, send data for dataset collection
  if (isRecording && buttonPressed) {
    sendToServer({ type: "record_frame", label: recordedLabel, frame: data });
  }
}

function updateRecognition(rec) {
  document.getElementById("recognized-char").textContent = rec.class;
  const pct = Math.round(rec.confidence * 100);
  document.getElementById("recognized-conf").textContent = `${pct}%`;
  const fill = document.getElementById("confidence-fill");
  fill.style.width = `${pct}%`;
  fill.style.background = rec.confidence >= 0.85
    ? "linear-gradient(90deg, #10b981, #4f6ef7)"
    : rec.confidence >= 0.6
      ? "linear-gradient(90deg, #f59e0b, #4f6ef7)"
      : "linear-gradient(90deg, #ef4444, #f59e0b)";
}

function updateSensorHUD(data) {
  if (data.raw_sensors) {
    const s = data.raw_sensors;
    if (s.s1) document.getElementById("s1-data").textContent = `A[${fv(s.s1.ax)}, ${fv(s.s1.ay)}, ${fv(s.s1.az)}]`;
    if (s.s2) document.getElementById("s2-data").textContent = `A[${fv(s.s2.ax)}, ${fv(s.s2.ay)}, ${fv(s.s2.az)}]`;
    if (s.s3) document.getElementById("s3-data").textContent = `A[${fv(s.s3.ax)}, ${fv(s.s3.ay)}, ${fv(s.s3.az)}]`;
  }
  if (data.fingertip) {
    const t = data.fingertip;
    document.getElementById("tip-data").textContent = `[${fv(t[0])}, ${fv(t[1])}, ${fv(t[2])}]`;
  }
}

function fv(v) { return typeof v === "number" ? v.toFixed(2) : "—"; }

// ── Training UI ──
let recordTimeout = null;
let isAutoRecordingEnabled = false;
let autoRecordArmed = false;

function toggleAutoRecord() {
   if (!isAutoRecordingEnabled) {
       const label = document.getElementById("train-label").value.trim();
       if (!label) { alert("글자를 입력하세요!"); return; }
       
       isAutoRecordingEnabled = true;
       autoRecordArmed = true; 
       document.getElementById("btn-record").textContent = "⏹ Disable Auto Record";
       document.getElementById("train-status").textContent = "Waiting for Button...";
   } else {
       isAutoRecordingEnabled = false;
       autoRecordArmed = false;
       isRecording = false;
       if (recordTimeout) clearTimeout(recordTimeout);
       sendToServer({ type: "stop_recording" });
       document.getElementById("btn-record").textContent = "⏺ Record (Auto)";
       document.getElementById("train-status").textContent = "Idle";
   }
}

function startDynamicRecording() {
  const label = document.getElementById("train-label").value.trim();
  if (!label) return;
  
  isRecording = true;
  recordedLabel = label;
  document.getElementById("train-status").textContent = "🔴 Recording Stroke...";
  sendToServer({ type: "start_recording", label });
  if (recordTimeout) { clearTimeout(recordTimeout); recordTimeout = null; }
}

function stopRecordingAndRearm() {
  isRecording = false;
  if (recordTimeout) { clearTimeout(recordTimeout); recordTimeout = null; }
  sendToServer({ type: "stop_recording" });
  setTimeout(() => sendToServer({ type: "get_sample_count" }), 500);
  
  if (isAutoRecordingEnabled) {
      document.getElementById("train-status").textContent = "✅ Saved. Waiting for next hit...";
      autoRecordArmed = true; // Wait for physical button to be pressed again
  } else {
      document.getElementById("train-status").textContent = "Saved.";
  }
}

// 오버라이드 (기존 HTML onclick 속성 대응)
window.startRecording = toggleAutoRecord;
window.stopRecording = toggleAutoRecord;

function startTraining() {
  if (isRecording) stopRecording();

  document.getElementById("train-status").textContent = "Training...";
  document.getElementById("btn-train").disabled = true;
  document.getElementById("btn-record").disabled = true;
  document.getElementById("train-label").disabled = true;

  // 가짜 로딩 바 대신 UI 초기화 및 실제 요청 전송 (웹소켓 응답에서 게이지와 완료 처리)
  document.getElementById("train-progress-fill").style.width = "5%";
  sendToServer({ type: "start_training", duration_sec: 3 });
}

// ── Controls ──
function deleteLastChar() {
  eraseLastChar();
}

function eraseLastChar() {
  const charEl = document.getElementById("recognized-char");
  const currentText = charEl.textContent;
  if (currentText.length > 0 && currentText !== '—') {
    const newText = currentText.slice(0, -1);
    charEl.textContent = newText || '—';
    dummySentence = newText;
    
    // 서버에도 삭제 명령 전송
    sendToServer({ type: "erase_last_char" });
    
    // 상태 뱅지 반짝임
    const badge = document.getElementById("writing-indicator");
    badge.querySelector(".writing-badge").textContent = "🔙 ERASE";
    badge.classList.add("active");
    setTimeout(() => {
      badge.classList.remove("active");
      badge.querySelector(".writing-badge").textContent = "✏️ WRITING";
    }, 300);
  }
}

function clearAllText() {
  document.getElementById("recognized-char").textContent = '—';
  document.getElementById("recognized-conf").textContent = '0%';
  dummySentence = "";
  sendToServer({ type: "clear_all_text" });
  
  const badge = document.getElementById("writing-indicator");
  badge.querySelector(".writing-badge").textContent = "🗑️ CLEARED";
  badge.classList.add("active");
  setTimeout(() => {
    badge.classList.remove("active");
    badge.querySelector(".writing-badge").textContent = "✏️ WRITING";
  }, 400);
}

function toggleRecogPanel() {
  const panel = document.getElementById("recognition-panel");
  const btn = document.getElementById("btn-toggle-recog");
  panel.classList.toggle("collapsed");
  btn.textContent = panel.classList.contains("collapsed") ? '▶' : '▼';
}

function clearTrajectory() {
  liveCount = 0;
  liveGeom.setDrawRange(0, 0);
  livePositions.fill(0);
  liveGeom.attributes.position.needsUpdate = true;
  liveCylMesh.count = 0;
  liveCylMesh.instanceMatrix.needsUpdate = true;
  morphMat.opacity = 0;
  morphParticleMat.opacity = 0;
}

let isFirstPersonView = true;
function toggleView() {
  isFirstPersonView = !isFirstPersonView;
  if (isFirstPersonView) {
    // 1인칭 시점 복귀 (원근감 최소화 및 하단 포커스 정면 뷰)
    camera.position.set(0, -0.05, 3.5);
    controls.target.set(0, -0.05, -0.6); 
    controls.maxAzimuthAngle = Math.PI / 16;
    controls.minAzimuthAngle = -Math.PI / 16;
    controls.maxPolarAngle = Math.PI / 2 + Math.PI / 16;
    controls.minPolarAngle = Math.PI / 2 - Math.PI / 16;
    controls.enableRotate = false;
    controls.enablePan = false;
  } else {
    // 3인칭 자유 시점 (마우스 제어 허용)
    camera.position.set(0.6, 0.4, 0.8);
    controls.target.set(0, 0, -0.3);
    controls.maxAzimuthAngle = Infinity;
    controls.minAzimuthAngle = -Infinity;
    controls.maxPolarAngle = Math.PI;
    controls.minPolarAngle = 0;
    controls.enableRotate = true; // 마우스 드래그로 화면 회전 가능
    controls.enablePan = true;    // 우클릭 팬 가능
  }
  controls.update();
}

let gridVisible = true;
function toggleGrid() {
  gridVisible = !gridVisible;
  gridHelper.visible = gridVisible;
  axesHelper.visible = gridVisible;
  ground.visible = gridVisible;
}

// ── 테스트 시뮬레이션 (T키 연타로 ABC 입력 테스트) ──
const dummyChars = ['A', 'B', 'C', ' ', 'A', 'i', 'r', 'W', 'r', 'i', 't', 'i', 'n', 'g'];
let dummyIdx = 0;
let dummySentence = "";
function simulateTyping() {
  const char = dummyChars[dummyIdx % dummyChars.length];
  dummyIdx++;
  dummySentence += char;
  
  // UI 업데이트 시뮬레이션
  document.getElementById("recognized-char").textContent = dummySentence;
  document.getElementById("recognized-conf").textContent = `Latest: '${char}'`;
  
  // 상태 뱃지 반짝임 효과
  const badge = document.getElementById("writing-indicator");
  badge.classList.add("active");
  setTimeout(() => badge.classList.remove("active"), 200);
}

function toggleConnection() {
  if (wsConnected && ws) ws.close();
  else connectWebSocket();
}

function toggleDashboard() {
  const overlay = document.getElementById("dashboard-overlay");
  if (overlay.style.display === "none") {
    overlay.style.display = "flex";
    initDashboard();
  } else {
    overlay.style.display = "none";
    if (activeSimulation) clearInterval(activeSimulation);
  }
}

// ── Dashboard & Chart.js Logic ──
let accuracyChart = null;
let activeSimulation = null;
const MOCK_CLASSES = [
  { word: "HELLO", samples: 120, bestAcc: 0.94 },
  { word: "WORLD", samples: 95, bestAcc: 0.88 },
  { word: "AIR", samples: 210, bestAcc: 0.98 },
  { word: "WRITING", samples: 154, bestAcc: 0.91 },
  { word: "ERASE", samples: 50, bestAcc: 0.75 }
];

function initDashboard() {
  // 서버에서 실제 데이터셋 정보 요청
  sendToServer({ type: "get_dataset_info" });
  // 실제 데이터가 있으면 그것을 표시, 없으면 MOCK 표시
  if (Object.keys(classHistoryData).length > 0) {
    renderRealClassList();
  } else {
    renderClassList();
  }
  if (!accuracyChart) initChart();
}

function renderClassList() {
  const container = document.getElementById("class-list");
  container.innerHTML = "";
  
  MOCK_CLASSES.forEach((cls, idx) => {
    const div = document.createElement("div");
    div.className = "class-item";
    div.innerHTML = `
      <span>${cls.word}</span>
      <span>${cls.samples}</span>
      <span>${(cls.bestAcc * 100).toFixed(1)}%</span>
    `;
    div.onclick = () => selectClass(cls, div);
    container.appendChild(div);
  });
}

function selectClass(cls, element) {
  // Update Highlights
  document.querySelectorAll(".class-item").forEach(el => el.classList.remove("selected"));
  element.classList.add("selected");
  
  document.getElementById("chart-title").textContent = `Real-time Accuracy Growth: ${cls.word}`;
  
  // Reset and Simulate real-time graph
  // Mock 데이터 시뮬레이션 (실제 학습은 processTrainingMetrics에서 처리)
  if (accuracyChart) {
    accuracyChart.data.labels = [];
    accuracyChart.data.datasets[0].data = [];
    accuracyChart.update();
  }
}

function initChart() {
  const ctx = document.getElementById("accuracyChart").getContext("2d");
  accuracyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [], // Epochs
      datasets: [{
        label: 'Accuracy (%)',
        data: [],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 3,
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#10b981',
        pointRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 }, // Disable animation for real-time speed
      scales: {
        y: { beginAtZero: false, suggestedMin: 0, suggestedMax: 100, grid: { color: 'rgba(15, 23, 42, 0.1)' } },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// ── Real PyTorch Data Integration ──
// This object stores real-time history for each class as it arrives
let classHistoryData = {};
let currentlySelectedClass = null;
// 서버에서 받은 실제 데이터셋 정보를 기반으로 Dashboard 클래스 목록을 렌더링
let datasetClasses = []; // [{word, samples, bestAcc}]

function renderRealClassList() {
  const container = document.getElementById("class-list");
  container.innerHTML = "";
  
  const sources = datasetClasses.length > 0 ? datasetClasses : 
    Object.entries(classHistoryData).map(([word, d]) => ({
      word, samples: d.samples || 0, bestAcc: d.bestAcc || 0
    }));
  
  if (sources.length === 0) {
    container.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 12px;">No data recorded yet.<br>Record samples first!</div>';
    return;
  }
  
  sources.forEach(cls => {
    const div = document.createElement("div");
    div.className = "class-item";
    const isSelected = currentlySelectedClass === cls.word ? " selected" : "";
    div.className = "class-item" + isSelected;
    div.innerHTML = `
      <span>${cls.word}</span>
      <span>${cls.samples}</span>
      <span>${(cls.bestAcc * 100).toFixed(1)}%</span>
    `;
    div.onclick = () => selectRealClass(cls.word);
    container.appendChild(div);
  });
}

function updateDashboardWithDatasetInfo(data) {
  // data = { type: "dataset_info", classes: [{name: "A", count: 5}, ...], total: 10 }
  if (data.classes && Array.isArray(data.classes)) {
    datasetClasses = data.classes.map(c => ({
      word: c.name,
      samples: c.count,
      bestAcc: (classHistoryData[c.name] && classHistoryData[c.name].bestAcc) 
        ? classHistoryData[c.name].bestAcc / 100 : 0,
    }));
  }
  
  // 총 샘플 수 업데이트
  if (data.total !== undefined) {
    document.getElementById("sample-count").textContent = `${data.total} samples`;
  }
  
  // Dashboard가 열려있으면 즉시 갱신
  const overlay = document.getElementById("dashboard-overlay");
  if (overlay.style.display !== "none") {
    renderRealClassList();
  }
}

function processTrainingMetrics(data) {
  // data.classes is a dict: { "A": { acc: 0.8, samples: 10 }, "B": ... }
  
  // 1. Update our local history and class list
  const classesObj = data.classes || {};
  let listHtml = "";
  
  for (const [word, stats] of Object.entries(classesObj)) {
    // init history array if needed
    if (!classHistoryData[word]) {
      classHistoryData[word] = { history: [], samples: stats.samples, bestAcc: 0 };
    }
    
    const accPct = stats.acc * 100;
    classHistoryData[word].history.push({ epoch: data.epoch, acc: accPct });
    if (accPct > classHistoryData[word].bestAcc) {
      classHistoryData[word].bestAcc = accPct;
    }
    classHistoryData[word].samples = stats.samples;
    
    // Append to UI list dynamically
    const isSelected = currentlySelectedClass === word ? "selected" : "";
    listHtml += `
      <div class="class-item ${isSelected}" onclick="selectRealClass('${word}')">
        <span>${word}</span>
        <span>${stats.samples}</span>
        <span>${classHistoryData[word].bestAcc.toFixed(1)}%</span>
      </div>
    `;
  }
  
  document.getElementById("class-list").innerHTML = listHtml;
  
  // 2. If a class is currently selected, update the graph points
  if (currentlySelectedClass && classHistoryData[currentlySelectedClass]) {
    updateChartForClass(currentlySelectedClass);
  } else if (!currentlySelectedClass && Object.keys(classesObj).length > 0) {
    // Select first available class automatically
    const firstClass = Object.keys(classesObj)[0];
    selectRealClass(firstClass);
  }
}

function selectRealClass(word) {
  currentlySelectedClass = word;
  
  // Update Highlight via re-render (handled next update) or manual class toggling
  document.querySelectorAll(".class-item").forEach(el => {
    if (el.querySelector("span").textContent === word) el.classList.add("selected");
    else el.classList.remove("selected");
  });
  
  document.getElementById("chart-title").textContent = `Real-time Accuracy Growth: ${word}`;
  updateChartForClass(word);
}

function updateChartForClass(word) {
  if (!accuracyChart || !classHistoryData[word]) return;
  
  const history = classHistoryData[word].history;
  
  // Reset Data and load from history
  accuracyChart.data.labels = history.map(h => `Ep ${h.epoch}`);
  accuracyChart.data.datasets[0].data = history.map(h => h.acc);
  
  // Keep sliding window if too many points (e.g. max 50 points)
  if (accuracyChart.data.labels.length > 50) {
    accuracyChart.data.labels = accuracyChart.data.labels.slice(-50);
    accuracyChart.data.datasets[0].data = accuracyChart.data.datasets[0].data.slice(-50);
  }
  
  accuracyChart.update();
}

document.addEventListener("keydown", (e) => {
  // 입력 폼에 포커스되어 있을 때는 단축키 무시
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  
  switch (e.key.toLowerCase()) {
    case "c": clearTrajectory(); break;
    case "r": toggleView(); break;
    case "g": toggleGrid(); break;
    case "z": recenterView(); break;
    case " ": e.preventDefault(); toggleConnection(); break;
    case "d": toggleDashboard(); break;
    case "t": simulateTyping(); break;
    case "backspace": e.preventDefault(); eraseLastChar(); break;
  }
});

// WS message handler (server → client)
function handleServerMessage(data) {
  if (data.type === "sample_count") {
    document.getElementById("sample-count").textContent = `${data.count} samples`;
  }
  else if (data.type === "training_metrics_per_class") {
    // 진행도 바 업데이트 (에포크 기반 실제 진행률)
    const pct = Math.min((data.epoch / data.total_epochs) * 100, 100);
    document.getElementById("train-progress-fill").style.width = `${pct}%`;
    processTrainingMetrics(data);
  }
  else if (data.type === "training_complete") {
    document.getElementById("train-status").textContent = `Done! Acc: ${(data.accuracy * 100).toFixed(0)}%`;
    document.getElementById("btn-train").disabled = false;
    document.getElementById("btn-record").disabled = false;
    document.getElementById("train-label").disabled = false;
    document.getElementById("train-progress-fill").style.width = "100%";
    setTimeout(() => { document.getElementById("train-progress-fill").style.width = "0%"; }, 2000);
    alert(`Training Custom Model Complete!\nAccuracy: ${(data.accuracy * 100).toFixed(1)}%\nPlease Restart Server. \n\n안드로이드 앱 연결 시, python main.py 서버를 반드시 재시작하여 새 모델을 로드해주세요.`);
  }
  else if (data.type === "training_error") {
    document.getElementById("train-status").textContent = `Error: Data Not Enough`;
    document.getElementById("btn-train").disabled = false;
    document.getElementById("btn-record").disabled = false;
    document.getElementById("train-label").disabled = false;
    document.getElementById("train-progress-fill").style.width = "0%";
    alert(data.message);
  }
  else if (data.type === "streaming_text") {
    document.getElementById("recognized-char").textContent = data.sentence || "—";
    document.getElementById("recognized-conf").textContent = `Latest: '${data.latest_char}'`;
  }
  else if (data.type === "dataset_info") {
    // 서버에서 받은 실제 데이터셋 정보로 Dashboard 업데이트
    updateDashboardWithDatasetInfo(data);
  }
}

// ── Demo mode ──
let demoAngle = 0;
function runDemo() {
  if (wsConnected) return;
  demoAngle += 0.02;
  const t = demoAngle;

  // 전완이 자유롭게 움직임 (고정 해제됨)
  const forearmPos = [Math.sin(t * 0.3) * 0.03, Math.cos(t * 0.2) * 0.02, Math.sin(t * 0.15) * 0.02];
  const handPos = [forearmPos[0] + Math.sin(t * 0.5) * 0.04, forearmPos[1] - 0.25 + Math.sin(t) * 0.02, forearmPos[2] + Math.cos(t * 0.3) * 0.03];
  const fingerPos = [handPos[0] + Math.sin(t) * 0.03, handPos[1] - 0.18 + Math.cos(t * 1.5) * 0.02, handPos[2] + Math.sin(t * 0.7) * 0.02];
  const tipPos = [fingerPos[0] + Math.sin(t * 1.2) * 0.015, fingerPos[1] - 0.08, fingerPos[2] + Math.cos(t) * 0.01];

  // 데모 좌표의 기본 중심점(Y ≒ -0.51)을 빼서 움직임만 추출한 뒤 감도를 곱함
  // 화면 중앙(Y = -0.25)에 위치하도록 최종 보정
  const dX = tipPos[0];
  const dY = tipPos[1] - (-0.51); 
  const dZ = tipPos[2];

  const tipX = dX * SENSITIVITY;
  const tipY = -0.25 + (dY * SENSITIVITY);
  const tipZ = -0.4 + (dZ * SENSITIVITY * 0.5);

  joints.forearm.position.set(
      (forearmPos[0]) * SENSITIVITY, 
      -0.25 + (forearmPos[1] - (-0.51)) * SENSITIVITY, 
      -0.4 + forearmPos[2]
  );
  joints.hand.position.set(
      (handPos[0]) * SENSITIVITY, 
      -0.25 + (handPos[1] - (-0.51)) * SENSITIVITY, 
      -0.4 + handPos[2]
  );
  joints.finger.position.set(
      (fingerPos[0]) * SENSITIVITY, 
      -0.25 + (fingerPos[1] - (-0.51)) * SENSITIVITY, 
      -0.4 + fingerPos[2]
  );
  fingertipMesh.position.set(tipX, tipY, tipZ);

  // 1인칭 뷰일 때 펜 숨김
  fingertipMesh.visible = !isFirstPersonView;

  // 투영 섀도우 X,Y 동기화
  projectionMesh.position.x = tipX;
  projectionMesh.position.y = tipY;

  updateBone(bones.forearm_hand, joints.forearm.position, joints.hand.position);
  updateBone(bones.hand_finger, joints.hand.position, joints.finger.position);
  updateBone(bones.finger_tip, joints.finger.position, fingertipMesh.position);

  // Z축을 -0.68로 강제 평면화하여 궤적 그리기
  const isWriting = Math.sin(t * 2) > 0;
  if (isWriting) {
    addLivePoint(tipX, tipY, -0.68);
    projectionMat.opacity = 0.8;
    projectionMat.color.setHex(0xef4444);
  } else {
    projectionMat.opacity = 0.15;
    projectionMat.color.setHex(0x000000); 
  }
  
  document.getElementById("tip-data").textContent = `[${tipX.toFixed(3)}, ${tipY.toFixed(3)}, ${tipZ.toFixed(3)}]`;
}
setInterval(runDemo, 16);

// ── Animation loop ──
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateMorph();

  // Tip glow
  const time = performance.now() * 0.003;
  fingertipMesh.material.emissiveIntensity = 0.15 + 0.15 * Math.sin(time);

  // FPS
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    document.getElementById("fps-value").textContent = fpsCounter;
    fpsCounter = 0;
    lastFpsTime = now;
  }
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
setTimeout(connectWebSocket, 1000);
