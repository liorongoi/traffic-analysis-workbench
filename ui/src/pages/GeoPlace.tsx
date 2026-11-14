import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api";
import FolderPicker from "../components/FolderPicker";
import { fmtT } from "./Events";

// A fixed camera on the map. Pick a map file on this computer (a GeoTIFF, or a picture with its corners), then click
// the same spot on the frame and on the map, four times or more. Each point's error when it is left out shows as you
// go, and a point the others disagree with is marked, so a wrong click shows before saving. Once saved, every counted
// target of the video is drawn on the map along the road it drove. The camera is kept by folder, like its lines.

type MapFile = { name: string; path: string; size: number; geotiff: boolean };
type MapInfo = { width: number; height: number; top: number; left: number; bottom: number; right: number; m_per_px: number };
type Pt = [number, number, number, number];                    // x, y on the frame; lat, lon on the map
type Fit = { residual_m: number[]; leave_one_out_m: number[]; suspect: number | null; rms_m: number };
type Geo = { points: Pt[]; map_path: string | null; corners: number[] | null; fit: Fit; author: string; created_at: string };
type MPath = { event_id: number; key: string; cls: string; first_t: number; last_t: number; path: [number, number, number][]; length_m: number };

const CLS_COLOUR: Record<string, string> = { car: "#4aa3e8", truck: "#e8a34a", bus: "#c86ae8", motorcycle: "#e84a6a",
  bicycle: "#6ae88a", person: "#e8e04a" };
const colour = (c: string) => CLS_COLOUR[c] ?? "#9fb3c8";

export default function GeoPlace({ missionId, versionId }: { missionId: string; versionId: number }) {
  const [rec, setRec] = useState<{ width: number; height: number; duration: number; rel_path: string } | null>(null);
  const [t, setT] = useState(1);
  const [folder, setFolder] = useState<string[]>([]);
  const [files, setFiles] = useState<MapFile[]>([]);
  const [mapPath, setMapPath] = useState<string | null>(null);
  const [corners, setCorners] = useState("");
  const [info, setInfo] = useState<MapInfo | null>(null);
  const [pts, setPts] = useState<Pt[]>([]);
  const [half, setHalf] = useState<[number, number] | null>(null);   // a frame click waiting for its map click
  const [fit, setFit] = useState<Fit | null>(null);
  const [saved, setSaved] = useState<Geo | null>(null);
  const [paths, setPaths] = useState<MPath[]>([]);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  useEffect(() => {
    api(`/missions/${missionId}/recordings/${versionId}/overlay`).then((o) => setRec({
      width: o.recording.width, height: o.recording.height, duration: o.recording.duration_s ?? 10, rel_path: o.recording.rel_path,
    })).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message }));
    api(`/missions/${missionId}/recordings/${versionId}/geo`).then((r) => {
      const g: Geo | null = r.geo;
      setSaved(g);
      if (g) {
        setPts(g.points); setFit(g.fit);
        if (g.map_path) { setMapPath(g.map_path); setFolder([g.map_path.replace(/[\\/][^\\/]*$/, "")]); setCorners(g.corners ? g.corners.join(",") : ""); }
      }
    }).catch(() => undefined);
  }, [missionId, versionId]);
  const loadPaths = () => api(`/missions/${missionId}/recordings/${versionId}/map-paths`).then((r) => setPaths(r.paths)).catch(() => setPaths([]));
  useEffect(() => { if (saved) loadPaths(); else setPaths([]); }, [saved]);

  useEffect(() => {
    if (!folder[0]) return;
    api(`/map-files?folder=${encodeURIComponent(folder[0])}`).then((r) => setFiles(r.files))
      .catch((e) => { setFiles([]); setMsg({ kind: "err", text: (e as ApiError).message }); });
  }, [folder[0]]);
  const mapQuery = mapPath ? `path=${encodeURIComponent(mapPath)}${corners.trim() ? `&corners=${encodeURIComponent(corners.trim())}` : ""}` : "";
  useEffect(() => {
    setInfo(null);
    if (!mapPath) return;
    api(`/map-info?${mapQuery}`).then(setInfo).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message }));
  }, [mapQuery]);

  // Fit as soon as there are four points (nothing is saved until Save).
  useEffect(() => {
    if (pts.length < 4) { setFit(null); return; }
    api(`/missions/${missionId}/recordings/${versionId}/geo`, { method: "PUT", body: { points: pts, check_only: true } })
      .then((r) => setFit(r.fit)).catch((e) => { setFit(null); setMsg({ kind: "err", text: (e as ApiError).message }); });
  }, [pts]);

  const save = async () => {
    try {
      const r = await api(`/missions/${missionId}/recordings/${versionId}/geo`, { method: "PUT",
        body: { points: pts, map_path: mapPath, corners: corners.trim() ? corners.split(",").map(Number) : null } });
      setSaved(r.geo);
      setMsg({ kind: "ok", text: "The camera is on the map. Every video from the same folder uses it." });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const clear = async () => {
    await api(`/missions/${missionId}/recordings/${versionId}/geo`, { method: "DELETE" });
    setSaved(null); setPts([]); setFit(null); setMsg({ kind: "ok", text: "The camera is no longer on the map." });
  };

  const onFrame = (x: number, y: number) => { setHalf([x, y]); setMsg(null); };
  const onMap = (lat: number, lon: number) => {
    if (!half) { setMsg({ kind: "err", text: "Click the spot on the frame first, then the same spot on the map." }); return; }
    setPts([...pts, [Math.round(half[0]), Math.round(half[1]), lat, lon]]);
    setHalf(null);
  };

  return (
    <div className="page grid" style={{ gap: 12, maxWidth: "none" }}>
      <div className="row" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <div>
          <a href={`#/m/${missionId}/events/play-${versionId}`}>← Back to the player</a>
          <h2 style={{ margin: "4px 0" }}>Put the camera on the map</h2>
          <div className="muted small">{rec?.rel_path}</div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {saved && <button onClick={clear}>Take it off the map</button>}
          <button className="primary" disabled={!fit || !mapPath || pts.length < 4} onClick={save}
            title="Saved for every video in this camera's folder">Save</button>
        </div>
      </div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}

      <div className="card grid" style={{ gap: 8 }}>
        <strong className="small">1. The map</strong>
        <FolderPicker label="Folder with the map file" value={folder} onChange={setFolder} placeholder="Folder with an orthophoto or satellite picture" />
        {files.length > 0 && <select aria-label="Map file" value={mapPath ?? ""} onChange={(e) => { setMapPath(e.target.value || null); setPts([]); }}>
          <option value="">Pick a map file…</option>
          {files.map((f) => <option key={f.path} value={f.path}>{f.name}{f.geotiff ? "" : " (needs its corners)"}</option>)}
        </select>}
        {folder[0] && files.length === 0 && <div className="muted small">No GeoTIFF, JPEG or PNG in that folder.</div>}
        {mapPath && !mapPath.match(/\.tiff?$/i) && <label className="small">Corners: top latitude, left longitude, bottom latitude, right longitude
          <input value={corners} onChange={(e) => setCorners(e.target.value)} placeholder="32.0805,34.7795,32.0725,34.7875" /></label>}
        {info && <div className="muted small">{info.width}×{info.height} pixels, {info.m_per_px} m a pixel, from {info.top.toFixed(5)}°, {info.left.toFixed(5)}° to {info.bottom.toFixed(5)}°, {info.right.toFixed(5)}°</div>}
      </div>

      <div className="card grid" style={{ gap: 8 }}>
        <strong className="small">2. The same spot on the frame, then on the map, four times or more</strong>
        <div className="muted small">Pick spots on the ground that you can find on both: road marking corners, kerb corners, manhole covers.
          Spread them across the road, near and far. Six or more is better than four.</div>
        <div className="row" style={{ gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 420px", minWidth: 300 }}>
            {rec && <FramePanel missionId={missionId} versionId={versionId} rec={rec} t={t} pts={pts} half={half} fit={fit}
              paths={[]} onPick={onFrame} />}
            {rec && <label className="small row" style={{ gap: 8 }}>Frame at {fmtT(t)}
              <input type="range" min={0} max={Math.max(0.1, rec.duration - 0.1)} step={0.1} value={t} onChange={(e) => setT(Number(e.target.value))} style={{ flex: 1 }} /></label>}
          </div>
          <div style={{ flex: "1 1 420px", minWidth: 300 }}>
            {info && mapPath ? <MapPanel src={`/api/map-image?${mapQuery}`} info={info} pts={pts} fit={fit} paths={paths} onPick={onMap} waiting={!!half} />
              : <div className="muted small" style={{ padding: 24 }}>Pick a map file first.</div>}
          </div>
        </div>
        {pts.length > 0 && <table className="small">
          <thead><tr><th>#</th><th>Frame</th><th>Map</th><th>Error</th><th>Left out</th><th /></tr></thead>
          <tbody>{pts.map((p, i) => <tr key={i} style={fit?.suspect === i ? { color: "var(--bad, #e84a6a)" } : undefined}>
            <td>{i + 1}</td><td className="mono">{p[0]}, {p[1]}</td><td className="mono">{p[2].toFixed(6)}, {p[3].toFixed(6)}</td>
            <td>{fit ? `${fit.residual_m[i]} m` : "–"}</td>
            <td>{fit?.leave_one_out_m[i] != null ? `${fit.leave_one_out_m[i]} m` : "–"}{fit?.suspect === i ? " · probably a wrong click" : ""}</td>
            <td><button onClick={() => setPts(pts.filter((_, k) => k !== i))} aria-label={`Remove point ${i + 1}`}>✕</button></td>
          </tr>)}</tbody>
        </table>}
        {half && <div className="muted small">Now click the same spot on the map.</div>}
        {fit && <div className="small">Fit: {fit.rms_m} m on average over the points
          {fit.leave_one_out_m.length > 0 && <>; a new point would be about {median(fit.leave_one_out_m)} m off</>}.</div>}
      </div>

      {saved && <div className="card grid" style={{ gap: 6 }}>
        <strong className="small">3. Targets on the map</strong>
        <div className="muted small">{paths.length} counted targets of this video, drawn where they touched the road (the bottom middle of each box).
          Total distance {Math.round(paths.reduce((a, p) => a + p.length_m, 0))} m.</div>
        <div className="row small" style={{ gap: 12, flexWrap: "wrap" }}>
          {[...new Set(paths.map((p) => p.cls))].map((c) => <span key={c}><span style={{ color: colour(c) }}>━</span> {c}</span>)}
        </div>
      </div>}
    </div>
  );
}

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0; };

function FramePanel({ missionId, versionId, rec, t, pts, half, fit, onPick }: {
  missionId: string; versionId: number; rec: { width: number; height: number }; t: number; pts: Pt[];
  half: [number, number] | null; fit: Fit | null; paths: MPath[]; onPick: (x: number, y: number) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const click = (e: React.MouseEvent) => {
    const s = svg.current!;
    const p = s.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const q = p.matrixTransform(s.getScreenCTM()!.inverse());
    onPick(Math.max(0, Math.min(rec.width, q.x)), Math.max(0, Math.min(rec.height, q.y)));
  };
  const r = Math.max(rec.width, rec.height) / 150;
  return (
    <svg ref={svg} viewBox={`0 0 ${rec.width} ${rec.height}`} style={{ width: "100%", cursor: "crosshair", background: "#000" }} onClick={click}
      role="img" aria-label="Frame of the camera: click a spot on the ground">
      <image href={`/api/missions/${missionId}/videos/${versionId}/frame?t=${t.toFixed(1)}&w=1920`} width={rec.width} height={rec.height} />
      {pts.map((p, i) => <Mark key={i} x={p[0]} y={p[1]} r={r} n={i + 1} bad={fit?.suspect === i} />)}
      {half && <Mark x={half[0]} y={half[1]} r={r} n={pts.length + 1} pending />}
    </svg>
  );
}

function Mark({ x, y, r, n, bad, pending }: { x: number; y: number; r: number; n: number; bad?: boolean; pending?: boolean }) {
  const c = bad ? "#e84a6a" : pending ? "#e8e04a" : "#4ae8c8";
  return <g>
    <circle cx={x} cy={y} r={r} fill="none" stroke={c} strokeWidth={r / 3} />
    <line x1={x - r * 1.6} y1={y} x2={x + r * 1.6} y2={y} stroke={c} strokeWidth={r / 5} />
    <line x1={x} y1={y - r * 1.6} x2={x} y2={y + r * 1.6} stroke={c} strokeWidth={r / 5} />
    <text x={x + r * 1.4} y={y - r * 1.4} fill={c} fontSize={r * 2.4} fontWeight={700}>{n}</text>
  </g>;
}

/** The map picture, zoomed with the wheel and moved by dragging; latitude and longitude are linear across it. */
function MapPanel({ src, info, pts, fit, paths, onPick, waiting }: {
  src: string; info: MapInfo; pts: Pt[]; fit: Fit | null; paths: MPath[]; onPick: (lat: number, lon: number) => void; waiting: boolean;
}) {
  const W = info.width, H = info.height;
  const [vb, setVb] = useState({ x: 0, y: 0, w: W, h: H });
  useEffect(() => setVb({ x: 0, y: 0, w: W, h: H }), [W, H, src]);
  // Start on the points already picked, or on the targets.
  useEffect(() => {
    const ll = pts.length ? pts.map((p) => [p[2], p[3]]) : paths.flatMap((p) => p.path.map((q) => [q[1], q[2]]));
    if (!ll.length) return;
    const xs = ll.map((q) => toX(q[1])), ys = ll.map((q) => toY(q[0]));
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const side = Math.max(x1 - x0, y1 - y0, 80) * 1.6;
    setVb({ x: (x0 + x1) / 2 - side / 2, y: (y0 + y1) / 2 - side / 2, w: side, h: side });
  }, [paths.length > 0, info]);
  const toX = (lon: number) => (lon - info.left) / (info.right - info.left) * W;
  const toY = (lat: number) => (info.top - lat) / (info.top - info.bottom) * H;
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const local = (e: { clientX: number; clientY: number }) => {
    const s = svg.current!;
    const p = s.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    return p.matrixTransform(s.getScreenCTM()!.inverse());
  };
  useEffect(() => {
    const el = svg.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const p = local(e);
      const k = Math.exp(Math.max(-120, Math.min(120, e.deltaY)) * 0.0025);
      setVb((v) => {
        const w = Math.min(W * 1.2, Math.max(20, v.w * k)), h = w * (v.h / v.w);
        return { x: p.x - (p.x - v.x) * (w / v.w), y: p.y - (p.y - v.y) * (h / v.h), w, h };
      });
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, [W]);
  const down = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY, moved: false }; };
  const move = (e: React.MouseEvent) => {
    const d = drag.current;
    if (!d) return;
    const s = svg.current!.getBoundingClientRect();
    const dx = (e.clientX - d.x) * vb.w / s.width, dy = (e.clientY - d.y) * vb.h / s.height;
    if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 3) d.moved = true;
    if (d.moved) { setVb((v) => ({ ...v, x: v.x - dx, y: v.y - dy })); d.x = e.clientX; d.y = e.clientY; }
  };
  const up = (e: React.MouseEvent) => {
    const d = drag.current; drag.current = null;
    if (d && !d.moved) {
      const p = local(e);
      onPick(info.top - p.y / H * (info.top - info.bottom), info.left + p.x / W * (info.right - info.left));
    }
  };
  const r = vb.w / 150;
  const lines = useMemo(() => paths.map((p) => ({ cls: p.cls, key: p.key,
    d: p.path.map((q, i) => `${i ? "L" : "M"}${toX(q[2]).toFixed(1)},${toY(q[1]).toFixed(1)}`).join(" ") })), [paths, info]);
  return (
    <div>
      <svg ref={svg} viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`} preserveAspectRatio="xMidYMid meet"
        style={{ width: "100%", aspectRatio: "1 / 1", background: "#0b1320", cursor: waiting ? "crosshair" : "grab" }}
        onMouseDown={down} onMouseMove={move} onMouseUp={up} onMouseLeave={() => { drag.current = null; }}
        role="img" aria-label="Map: wheel to zoom, drag to move, click the spot">
        <image href={src} width={W} height={H} />
        {lines.map((l, i) => <path key={i} d={l.d} fill="none" stroke={colour(l.cls)} strokeWidth={r / 2.5} strokeOpacity={0.85}>
          <title>{l.key} · {l.cls}</title></path>)}
        {pts.map((p, i) => <Mark key={i} x={toX(p[3])} y={toY(p[2])} r={r} n={i + 1} bad={fit?.suspect === i} />)}
      </svg>
      <div className="muted small">Wheel to zoom, drag to move.</div>
    </div>
  );
}
