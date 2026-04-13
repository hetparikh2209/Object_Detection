'use client';
import { useState, useCallback } from 'react';
import { loadModel, isModelLoaded } from '@/lib/yolo';

export function useYoloModel() {
  const [status, setStatus] = useState({
    loading:  false,
    loaded:   false,
    error:    null,
    progress: 0,
  });

  const load = useCallback(async () => {
    if (isModelLoaded()) {
      setStatus({ loading: false, loaded: true, error: null, progress: 100 });
      return;
    }

    setStatus({ loading: true, loaded: false, error: null, progress: 0 });

    try {
      await loadModel((p) =>
        setStatus((prev) => ({ ...prev, progress: p }))
      );
      setStatus({ loading: false, loaded: true, error: null, progress: 100 });
    } catch (err) {
      setStatus({ loading: false, loaded: false, error: err.message, progress: 0 });
    }
  }, []);

  return { status, load };
}
