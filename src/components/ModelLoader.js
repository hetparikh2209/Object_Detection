'use client';

export default function ModelLoader({ status, onLoad }) {
  if (status.loaded) {
    return (
      <div className="model-status model-ready">
        <span className="status-dot pulse" />
        YOLOv8n · Ready
      </div>
    );
  }

  if (status.loading) {
    return (
      <div className="model-status model-loading">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${status.progress}%` }} />
        </div>
        <span>Loading model… {status.progress}%</span>
      </div>
    );
  }

  if (status.error) {
    return (
      <div className="model-status model-error">
        ⚠ {status.error}
        <button onClick={onLoad} className="retry-btn">Retry</button>
      </div>
    );
  }

  return (
    <button className="btn-load-model" onClick={onLoad}>
      🧠 Load YOLOv8 Model
    </button>
  );
}
