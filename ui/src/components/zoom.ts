import { useCallback, useEffect, useRef, useState } from "react";

// Zoom and pan for an image stage. The stage holds a layer that is scaled from its top-left corner,
// so stage pixel p shows layer pixel (p - t) / z. The layer never leaves a gap at the stage edges.
export type View = { z: number; tx: number; ty: number };
export const MIN_ZOOM = 1;
export const MAX_ZOOM = 10;
const FIT: View = { z: 1, tx: 0, ty: 0 };

// `ready` turns true once the stage element is on the page.
export function useZoom(ready: boolean) {
  const stageRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>(FIT);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const sizeRef = useRef(size);
  sizeRef.current = size;

  const clamp = useCallback((v: View): View => {
    const { w, h } = sizeRef.current;
    const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.z));
    return { z, tx: Math.min(0, Math.max(w - w * z, v.tx)), ty: Math.min(0, Math.max(h - h * z, v.ty)) };
  }, []);

  // Zoom by a factor while keeping the stage point (cx, cy) still; with no point, keep the centre still.
  const zoomAt = useCallback((factor: number, cx?: number, cy?: number) => {
    setView((v) => {
      const { w, h } = sizeRef.current;
      const px = cx ?? w / 2, py = cy ?? h / 2;
      const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.z * factor));
      const k = z / v.z;
      return clamp({ z, tx: px - (px - v.tx) * k, ty: py - (py - v.ty) * k });
    });
  }, [clamp]);
  const panBy = useCallback((dx: number, dy: number) => setView((v) => clamp({ ...v, tx: v.tx + dx, ty: v.ty + dy })), [clamp]);
  // Centre the stage on layer point (lx, ly), in unscaled stage pixels.
  const centerOn = useCallback((lx: number, ly: number) => setView((v) => {
    const { w, h } = sizeRef.current;
    return clamp({ ...v, tx: w / 2 - lx * v.z, ty: h / 2 - ly * v.z });
  }), [clamp]);
  const fit = useCallback(() => setView(FIT), []);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight });
      setView((v) => clamp(v));
    });
    ro.observe(el);
    // Mouse wheel and trackpad pinch zoom around the pointer. Needs a non-passive listener to keep the page still.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? r.height : 1;
      const dy = Math.max(-120, Math.min(120, e.deltaY * unit));
      zoomAt(Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0022)), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => { ro.disconnect(); el.removeEventListener("wheel", onWheel); };
  }, [ready, zoomAt, clamp]);

  return { stageRef, view, size, zoomAt, panBy, centerOn, fit };
}
