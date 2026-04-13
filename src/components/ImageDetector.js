'use client';
import { useRef, useState, useCallback, useEffect } from 'react';
import { runInference, drawDetections } from '@/lib/yolo';

export default function ImageDetector({ modelLoaded }) {
  const [imageSrc,   setImageSrc]   = useState(null);
  const [detections, setDetections] = useState([]);
  const [inferring,  setInferring]  = useState(false);
  const [inferTime,  setInferTime]  = useState(null);

  const imgRef    = useRef(null);
  const canvasRef = useRef(null);

  // ── File helpers ────────────────────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    setImageSrc(URL.createObjectURL(file));
    setDetections([]);
    setInferTime(null);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('image/')) handleFile(file);
    },
    [handleFile]
  );

  const handleInput = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  // ── Clear image ─────────────────────────────────────────────────────────────
  const clearImage = useCallback(() => {
    setImageSrc(null);
    setDetections([]);
    setInferTime(null);
  }, []);

  // ── Sync overlay canvas dimensions ──────────────────────────────────────────
  const syncCanvas = useCallback(() => {
    const img    = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    canvas.width  = img.clientWidth;
    canvas.height = img.clientHeight;
    if (detections.length > 0) {
      drawDetections(
        canvas, detections,
        canvas.width  / img.naturalWidth,
        canvas.height / img.naturalHeight
      );
    }
  }, [detections]);

  useEffect(() => {
    window.addEventListener('resize', syncCanvas);
    return () => window.removeEventListener('resize', syncCanvas);
  }, [syncCanvas]);

  // ── Run YOLO ────────────────────────────────────────────────────────────────
  const runDetection = useCallback(async () => {
    if (!imgRef.current || !canvasRef.current || !imageSrc) return;
    setInferring(true);

    try {
      const img = imgRef.current;
      const t0  = performance.now();
      const dets = await runInference(img, img.naturalWidth, img.naturalHeight);
      setInferTime(performance.now() - t0);
      setDetections(dets);

      const canvas = canvasRef.current;
      canvas.width  = img.clientWidth;
      canvas.height = img.clientHeight;
      drawDetections(
        canvas, dets,
        canvas.width  / img.naturalWidth,
        canvas.height / img.naturalHeight
      );
    } finally {
      setInferring(false);
    }
  }, [imageSrc]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="detector-wrapper">
      {/* Drop zone / preview */}
      <label
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <input
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={handleInput}
        />

        {!imageSrc ? (
          <div className="drop-zone-inner">
            <div className="drop-icon">📷</div>
            <p className="drop-title">Drop an image here</p>
            <p className="drop-sub">or click to browse — JPG, PNG, WebP</p>
          </div>
        ) : (
          <div className="preview-container">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imgRef}
              src={imageSrc}
              alt="uploaded"
              className="preview-img"
              onLoad={syncCanvas}
            />
            <canvas ref={canvasRef} className="overlay-canvas" />
          </div>
        )}
      </label>

      {/* Controls */}
      <div className="controls-row">
        {imageSrc && (
          <button
            className="btn-detect"
            onClick={runDetection}
            disabled={!modelLoaded || inferring}
          >
            {inferring ? <span className="spinner" /> : '⚡ Detect'}
          </button>
        )}

        {imageSrc && (
          <button
            className="btn-clear"
            onClick={clearImage}
            title="Remove uploaded image"
          >
            ✕ Clear
          </button>
        )}

        {inferTime !== null && (
          <span className="stat-badge">
            🕒 {inferTime.toFixed(0)} ms &nbsp;·&nbsp; {detections.length} objects
          </span>
        )}
      </div>

      {/* Detection chips */}
      {detections.length > 0 && (
        <div className="detections-grid">
          {detections.map((d, i) => (
            <div key={i} className="det-chip">
              <span className="det-label">{d.label}</span>
              <span className="det-score">{(d.score * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
