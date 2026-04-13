import * as ort from 'onnxruntime-web';

// Point WASM files to Next.js public folder
ort.env.wasm.wasmPaths = '/';

// ── Target classes ──────────────────────────────────────────────────────────
export const TARGET_CLASSES = [
  { id: 0,  label: 'Person',        color: '#FF6B6B', icon: '🚶' },
  { id: 2,  label: 'Car',           color: '#4ECDC4', icon: '🚗' },
  { id: 5,  label: 'Bus',           color: '#FFE66D', icon: '🚌' },
  { id: 7,  label: 'Truck',         color: '#A8E6CF', icon: '🚛' },
  { id: 9,  label: 'Traffic Light', color: '#FF8B94', icon: '🚦' },
];

const TARGET_CLASS_IDS = new Set(TARGET_CLASSES.map((c) => c.id));
const CLASS_MAP = new Map(TARGET_CLASSES.map((c) => [c.id, c]));

const MODEL_INPUT_SIZE = 640;
const CONF_THRESHOLD   = 0.35;
const IOU_THRESHOLD    = 0.45;

let session = null;

// ── Load model ───────────────────────────────────────────────────────────────
export async function loadModel(onProgress) {
  if (session) return;

  onProgress?.(10);
  const res = await fetch('/yolov8n.onnx');
  onProgress?.(30);

  const buf = await res.arrayBuffer();
  onProgress?.(70);

  session = await ort.InferenceSession.create(new Uint8Array(buf), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  onProgress?.(100);
}

export function isModelLoaded() {
  return session !== null;
}

// ── Letterbox resize ─────────────────────────────────────────────────────────
function letterbox(canvas, ctx, source, srcW, srcH) {
  const scale = Math.min(MODEL_INPUT_SIZE / srcW, MODEL_INPUT_SIZE / srcH);
  const nw    = Math.round(srcW * scale);
  const nh    = Math.round(srcH * scale);
  const padX  = Math.floor((MODEL_INPUT_SIZE - nw) / 2);
  const padY  = Math.floor((MODEL_INPUT_SIZE - nh) / 2);

  canvas.width  = MODEL_INPUT_SIZE;
  canvas.height = MODEL_INPUT_SIZE;
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  ctx.drawImage(source, padX, padY, nw, nh);

  return { scale, padX, padY };
}

// ── RGB normalise to CHW Float32 ─────────────────────────────────────────────
function preprocess(canvas, ctx) {
  const { data } = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const A = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const f = new Float32Array(3 * A);
  for (let i = 0; i < A; i++) {
    f[i]       = data[i * 4]     / 255;
    f[i + A]   = data[i * 4 + 1] / 255;
    f[i + 2*A] = data[i * 4 + 2] / 255;
  }
  return f;
}

// ── IoU ───────────────────────────────────────────────────────────────────────
function iou(a, b) {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aArea = (a[2] - a[0]) * (a[3] - a[1]);
  const bArea = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (aArea + bArea - inter + 1e-6);
}

// ── Non-Maximum Suppression ───────────────────────────────────────────────────
function nms(boxes) {
  const sorted     = [...boxes].sort((a, b) => b.score - a.score);
  const keep       = [];
  const suppressed = new Set();

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      const a = sorted[i], b = sorted[j];
      if (iou([a.x1,a.y1,a.x2,a.y2], [b.x1,b.y1,b.x2,b.y2]) > IOU_THRESHOLD) {
        suppressed.add(j);
      }
    }
  }
  return keep;
}

// ── Run inference ─────────────────────────────────────────────────────────────
export async function runInference(source, srcW, srcH) {
  if (!session) throw new Error('Model not loaded');

  const canvas = document.createElement('canvas');
  const ctx    = canvas.getContext('2d');
  const { scale, padX, padY } = letterbox(canvas, ctx, source, srcW, srcH);

  const inputTensor = new ort.Tensor(
    'float32',
    preprocess(canvas, ctx),
    [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]
  );

  const feeds   = { [session.inputNames[0]]: inputTensor };
  const results = await session.run(feeds);
  const out     = results[session.outputNames[0]];

  // YOLOv8 output shape: [1, 84, 8400]
  const data = out.data;           // Float32Array
  const N    = out.dims[2];        // 8400 anchors
  const F    = out.dims[1];        // 84  (4 box + 80 classes)

  const detections = [];

  for (let i = 0; i < N; i++) {
    // Find the highest-scoring class
    let maxScore = 0, classId = -1;
    for (let c = 4; c < F; c++) {
      const s = data[c * N + i];
      if (s > maxScore) { maxScore = s; classId = c - 4; }
    }

    if (maxScore < CONF_THRESHOLD)        continue;
    if (!TARGET_CLASS_IDS.has(classId))   continue;

    const cx = data[0 * N + i];
    const cy = data[1 * N + i];
    const w  = data[2 * N + i];
    const h  = data[3 * N + i];

    // Map back to original image coords
    const x1 = ((cx - w / 2) - padX) / scale;
    const y1 = ((cy - h / 2) - padY) / scale;
    const x2 = ((cx + w / 2) - padX) / scale;
    const y2 = ((cy + h / 2) - padY) / scale;

    const cfg = CLASS_MAP.get(classId);
    detections.push({ x1, y1, x2, y2, score: maxScore, classId, label: cfg.label });
  }

  return nms(detections);
}

// ── Draw bounding boxes on canvas ────────────────────────────────────────────
export function drawDetections(canvas, detections, scaleX, scaleY) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  detections.forEach((det) => {
    const cfg = CLASS_MAP.get(det.classId);
    const color = cfg.color;

    const x1 = det.x1 * scaleX;
    const y1 = det.y1 * scaleY;
    const x2 = det.x2 * scaleX;
    const y2 = det.y2 * scaleY;
    const bw = x2 - x1;
    const bh = y2 - y1;

    // Box outline
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(x1, y1, bw, bh);

    // Corner accent lines
    const cl = Math.min(bw, bh) * 0.2;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x1,        y1 + cl); ctx.lineTo(x1, y1);        ctx.lineTo(x1 + cl, y1);
    ctx.moveTo(x2 - cl,   y1);      ctx.lineTo(x2, y1);        ctx.lineTo(x2,       y1 + cl);
    ctx.moveTo(x2,        y2 - cl); ctx.lineTo(x2, y2);        ctx.lineTo(x2 - cl,  y2);
    ctx.moveTo(x1 + cl,   y2);      ctx.lineTo(x1, y2);        ctx.lineTo(x1,       y2 - cl);
    ctx.stroke();

    // Label tag
    const labelText = `${cfg.icon} ${det.label} ${(det.score * 100).toFixed(0)}%`;
    ctx.font = 'bold 12px "Space Mono", monospace';
    const tw   = ctx.measureText(labelText).width;
    const tagH = 20;
    const tagY = y1 > tagH + 6 ? y1 - tagH - 4 : y1 + 4;

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x1, tagY, tw + 10, tagH, 4);
    ctx.fill();

    ctx.fillStyle = '#000';
    ctx.fillText(labelText, x1 + 5, tagY + 13);
  });
}
