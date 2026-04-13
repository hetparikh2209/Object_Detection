'use client';
import { useState } from 'react';
import { useYoloModel } from '@/hooks/useYoloModel';
import ModelLoader from '@/components/ModelLoader';
import ImageDetector from '@/components/ImageDetector';
import VideoDetector from '@/components/VideoDetector';
import { TARGET_CLASSES } from '@/lib/yolo';

export default function Home() {
  const { status, load } = useYoloModel();
  const [mode, setMode]  = useState('image');

  return (
    <main className="app-shell">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo-badge">
            <span>🚦</span>
          </div>
          <div>
            <h1 className="app-title">TrafficVision</h1>
            <p className="app-sub">Real-time object detection · YOLOv8n</p>
          </div>
        </div>
        <div className="header-right">
          <ModelLoader status={status} onLoad={load} />
        </div>
      </header>

      {/* ── Class legend ─────────────────────────────────────────────────── */}
      <div className="legend-bar">
        {TARGET_CLASSES.map((cls) => (
          <div key={cls.id} className="legend-chip">
            <span className="legend-dot" style={{ background: cls.color }} />
            {cls.icon} {cls.label}
          </div>
        ))}
      </div>

      {/* ── Mode tabs ────────────────────────────────────────────────────── */}
      <div className="tabs">
        <button
          className={`tab ${mode === 'image' ? 'tab-active' : ''}`}
          onClick={() => setMode('image')}
        >
          📷 Image
        </button>
        <button
          className={`tab ${mode === 'video' ? 'tab-active' : ''}`}
          onClick={() => setMode('video')}
        >
          🎬 Video
        </button>
      </div>

      {/* ── Detection panel ──────────────────────────────────────────────── */}
      <section className="panel">
        {!status.loaded && !status.loading && (
          <div className="model-hint">
            👆 Load the model above to begin detecting
          </div>
        )}
        {mode === 'image'
          ? <ImageDetector modelLoaded={status.loaded} />
          : <VideoDetector modelLoaded={status.loaded} />
        }
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="app-footer">
        Powered by YOLOv8n · ONNX Runtime Web · All inference runs in your browser
      </footer>

    </main>
  );
}
