'use client';
import { useRef, useState, useCallback, useEffect } from 'react';
import { runInference, drawDetections } from '@/lib/yolo';

export default function VideoDetector({ modelLoaded }) {
  const [videoSrc, setVideoSrc] = useState(null);
  const [running,  setRunning]  = useState(false);
  const [fps,      setFps]      = useState(0);
  const [counts,   setCounts]   = useState({});

  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const rafRef    = useRef(0);
  const lastTime  = useRef(0);
  const fpsArr    = useRef([]);

  // ── File helpers ────────────────────────────────────────────────────────────
  const handleFile = useCallback(
    (file) => {
      if (running) stopDetection();
      setVideoSrc(URL.createObjectURL(file));
      setCounts({});
      setFps(0);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [running]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('video/')) handleFile(file);
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

  // ── Frame loop ──────────────────────────────────────────────────────────────
  const processFrame = useCallback(async (now) => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended) return;

    // Sync canvas to display size
    canvas.width  = video.clientWidth;
    canvas.height = video.clientHeight;

    // FPS rolling average
    const dt = now - lastTime.current;
    lastTime.current = now;
    if (dt > 0) {
      fpsArr.current.push(1000 / dt);
      if (fpsArr.current.length > 10) fpsArr.current.shift();
      const avg = fpsArr.current.reduce((a, b) => a + b, 0) / fpsArr.current.length;
      setFps(Math.round(avg));
    }

    // Inference
    const dets = await runInference(video, video.videoWidth, video.videoHeight);
    drawDetections(
      canvas, dets,
      canvas.width  / video.videoWidth,
      canvas.height / video.videoHeight
    );

    // Per-class counts
    const c = {};
    dets.forEach((d) => { c[d.label] = (c[d.label] ?? 0) + 1; });
    setCounts(c);

    rafRef.current = requestAnimationFrame(processFrame);
  }, []);

  // ── Start / stop ────────────────────────────────────────────────────────────
  const startDetection = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.play();
    setRunning(true);
    lastTime.current = performance.now();
    rafRef.current   = requestAnimationFrame(processFrame);
  }, [processFrame]);

  const stopDetection = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    videoRef.current?.pause();
    setRunning(false);
    setFps(0);
  }, []);

  // ── Clear video ─────────────────────────────────────────────────────────────
  const clearVideo = useCallback(() => {
    stopDetection();
    setVideoSrc(null);
    setCounts({});
    setFps(0);
  }, [stopDetection]);

  // Cleanup on unmount
  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

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
          accept="video/*"
          className="sr-only"
          onChange={handleInput}
        />

        {!videoSrc ? (
          <div className="drop-zone-inner">
            <div className="drop-icon">🎬</div>
            <p className="drop-title">Drop a video here</p>
            <p className="drop-sub">or click to browse — MP4, WebM, MOV</p>
          </div>
        ) : (
          <div className="preview-container" onClick={(e) => e.preventDefault()}>
            <video
              ref={videoRef}
              src={videoSrc}
              className="preview-img"
              muted
              playsInline
            />
            <canvas ref={canvasRef} className="overlay-canvas" />
            {running && <div className="fps-badge">⚡ {fps} FPS</div>}
          </div>
        )}
      </label>

      {/* Controls */}
      <div className="controls-row">
        {videoSrc && (
          <button
            className={`btn-detect ${running ? 'btn-stop' : ''}`}
            onClick={running ? stopDetection : startDetection}
            disabled={!modelLoaded}
          >
            {running ? '⏹ Stop' : '▶ Start Detection'}
          </button>
        )}

        {videoSrc && (
          <button
            className="btn-clear"
            onClick={clearVideo}
            title="Remove uploaded video"
          >
            ✕ Clear
          </button>
        )}
      </div>

      {/* Per-class count chips */}
      {Object.keys(counts).length > 0 && (
        <div className="detections-grid">
          {Object.entries(counts).map(([label, count]) => (
            <div key={label} className="det-chip">
              <span className="det-label">{label}</span>
              <span className="det-score">×{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
