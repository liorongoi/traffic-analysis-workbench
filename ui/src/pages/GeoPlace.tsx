import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, post } from "../api";
import MapSource, { type Catalogue, type LatLon } from "../components/MapSources";
import { Choice } from "../components/Pick";
import { fmtT } from "./Events";

// A fixed camera on the map. Pick a map file on this computer (a GeoTIFF, or a picture with its corners), then click
// the same spot on the frame and on the map, four times or more. Each point's error when it is left out shows as you
// go, and a point the others disagree with is marked, so a wrong click shows before saving. Once saved, every counted
// target of the video is drawn on the map along the road it drove. The camera is kept by folder, like its lines.
// The map and the ground heights come from files, an online service or a server on our network (MapSources).
// The frame is laid on the map (no black edges) inside its footprint: the near part green when the camera is placed by
// points (red for flight data only), the far part, toward the horizon, in its own colour.

type MapInfo = { width: number; height: number; top: number; left: number; bottom: number; right: number; m_per_px: number };
type Pt = [number, number, number, number];                    // x, y on the frame; lat, lon on the map
type Fit = { residual_m: number[]; leave_one_out_m: number[]; suspect: number | null; rms_m: number;
  heights_m?: (number | null)[]; height_span_m?: number | null; not_flat?: boolean };
type Geo = { points: Pt[]; map_path: string | null; corners: number[] | null; dtm_path: string | null; fit: Fit; author: string; created_at: string };
type Platform = { lat?: number; lon?: number; alt_asl?: number; agl?: number; heading?: number; pitch?: number; roll?: number; hfov?: number;
  search_box?: [number, number, number, number] | null };
type Job = { state: "running" | "done" | "failed"; done: number; total: number | null; error: string | null };
type Outline = { t: number; how: string; registered: boolean; near: LatLon[] | null; far: LatLon[] | null };
type Track = { fps: number; made_at: string; seconds: number; matcher: string; summary: Record<string, number>; outlines: Outline[] };
const pct = (n: number, tr: Track) => `${Math.round(100 * n / Math.max(1, tr.outlines.length))}%`;
type DtmInfo = { name: string; min_m: number | null; max_m: number | null; m_per_sample: number; top: number; left: number; bottom: number; right: number };
export type Footprint = { how: "points" | "points-poor" | "flight-data" | "map-match" | "sky"; registered: boolean; note?: string | null;
  near: LatLon[] | null; far: LatLon[] | null; bounds: [number, number, number, number] | null;
  view_bounds?: [number, number, number, number]; horizon?: boolean; beyond?: boolean; frame_w?: number; frame_h?: number };
type MPath = { event_id: number; key: string; cls: string; first_t: number; last_t: number; path: [number, number, number][]; length_m: number };

const CLS_COLOUR: Record<string, string> = { car: "#4aa3e8", truck: "#e8a34a", bus: "#c86ae8", motorcycle: "#e84a6a",
  bicycle: "#6ae88a", person: "#e8e04a" };
const colour = (c: string) => CLS_COLOUR[c] ?? "#9fb3c8";

export default function GeoPlace({ missionId, versionId }: { missionId: string; versionId: number }) {
  const [rec, setRec] = useState<{ width: number; height: number; duration: number; rel_path: string } | null>(null);
  const [t, setT] = useState(1);
  const [cat, setCat] = useState<Catalogue | null>(null);
  const [home, setHome] = useState<LatLon | null>(null);
  const [mapPath, setMapPath] = useState<string | null>(null);
  const [dtmPath, setDtmPath] = useState<string | null>(null);
  const [dtm, setDtm] = useState<DtmInfo | null>(null);
  const [fp, setFp] = useState<Footprint | null>(null);
  const [layers, setLayers] = useState({ frame: true, footprint: true, shade: true });
  const [aligning, setAligning] = useState(false);
  const [corners, setCorners] = useState("");
  const [info, setInfo] = useState<MapInfo | null>(null);
  const [pts, setPts] = useState<Pt[]>([]);
  const [half, setHalf] = useState<[number, number] | null>(null);   // a frame click waiting for its map click
  const [fit, setFit] = useState<Fit | null>(null);
  const [saved, setSaved] = useState<Geo | null>(null);
  const [paths, setPaths] = useState<MPath[]>([]);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [plat, setPlat] = useState<Platform>({});
  const [tel, setTel] = useState<{ source: string | null } | null>(null);
  const [boxMode, setBoxMode] = useState<LatLon | "start" | null>(null);   // drawing the search square: its first corner
  const [prep, setPrep] = useState<{ job: Job | null; track: Track | null }>({ job: null, track: null });
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    api(`/missions/${missionId}/recordings/${versionId}/overlay`).then((o) => setRec({
      width: o.recording.width, height: o.recording.height, duration: o.recording.duration_s ?? 10, rel_path: o.recording.rel_path,
    })).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message }));
    api(`/missions/${missionId}/recordings/${versionId}/geo`).then((r) => {
      const g: Geo | null = r.geo;
      setSaved(g);
      if (g) {
        setPts(g.points); setFit(g.fit);
        if (g.map_path) { setMapPath(g.map_path); setCorners(g.corners ? g.corners.join(",") : ""); }
        if (g.dtm_path) setDtmPath(g.dtm_path);
      }
    }).catch(() => undefined);
    api(`/missions/${missionId}`).then((r) => { const c = r.config; if (c?.latitude != null && c?.longitude != null) setHome([c.latitude, c.longitude]); }).catch(() => undefined);
  }, [missionId, versionId]);
  const loadCat = () => api("/map-sources").then(setCat).catch(() => undefined);
  useEffect(() => {
    api(`/missions/${missionId}/recordings/${versionId}/platform`).then((r) => setPlat(r.platform ?? {})).catch(() => undefined);
    api(`/missions/${missionId}/recordings/${versionId}/telemetry`).then(setTel).catch(() => setTel(null));
  }, [missionId, versionId]);
  const loadPrep = () => mapPath ? api(`/missions/${missionId}/recordings/${versionId}/georeg?map=${encodeURIComponent(mapPath)}`).then(setPrep).catch(() => undefined) : Promise.resolve();
  useEffect(() => { loadPrep(); }, [mapPath]);
  useEffect(() => {                               // follow a preparation while it runs
    if (prep.job?.state !== "running") return;
    const id = setTimeout(loadPrep, 1000);
    return () => clearTimeout(id);
  }, [prep]);
  const savePlat = async (p: Platform) => {
    try {
      const r = await api(`/missions/${missionId}/recordings/${versionId}/platform`, { method: "PUT", body: p });
      setPlat(r.platform ?? {}); setT((x) => x + 0.0001);
      setMsg({ kind: "ok", text: "Platform data saved." });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const prepare = async () => {
    try {
      const r = await post(`/missions/${missionId}/recordings/${versionId}/georeg`, { map_path: mapPath, dtm_path: dtmPath,
        corners: corners.trim() ? corners.split(",").map(Number) : null });
      setPrep({ ...prep, job: r.job });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const trackFp = (tt: number): Footprint | null => {
    const o = prep.track?.outlines;
    if (!o?.length) return null;
    let best = o[0];
    for (const x of o) if (Math.abs(x.t - tt) < Math.abs(best.t - tt)) best = x;
    if (best.how === "sky") return { how: "sky", registered: false, near: null, far: null, bounds: null, note: SKY_NOTE };
    if (!best.near) return null;
    const a = [...best.near, ...best.far!];
    return { how: best.how as Footprint["how"], registered: best.registered, near: best.near, far: best.far, horizon: false,
      bounds: null, view_bounds: [Math.max(...a.map((q) => q[0])), Math.min(...a.map((q) => q[1])), Math.min(...a.map((q) => q[0])), Math.max(...a.map((q) => q[1]))] };
  };
  useEffect(() => { loadCat(); }, []);
  useEffect(() => {
    setDtm(null);
    if (!dtmPath) return;
    api(`/terrain-info?path=${encodeURIComponent(dtmPath)}`).then(setDtm).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message }));
  }, [dtmPath]);
  // Where the frame at the slider's time lies on the ground (points, or the recording's flight data).
  const dtmQ = (dtmPath ? `&dtm=${encodeURIComponent(dtmPath)}` : "") + (mapPath && info ? `&map=${encodeURIComponent(mapPath)}${corners.trim() ? `&corners=${encodeURIComponent(corners.trim())}` : ""}` : "");
  useEffect(() => {
    const id = setTimeout(() => api(`/missions/${missionId}/recordings/${versionId}/footprint?t=${t.toFixed(1)}${dtmQ}`)
      .then((r) => setFp(r.footprint)).catch(() => setFp(null)).finally(() => setAligning(false)), 250);
    return () => clearTimeout(id);
  }, [t, saved, dtmQ, prep.track?.made_at, plat]);
  useEffect(() => { setAligning(!!mapPath); }, [t, dtmQ]);
  const loadPaths = () => api(`/missions/${missionId}/recordings/${versionId}/map-paths`).then((r) => setPaths(r.paths)).catch(() => setPaths([]));
  useEffect(() => { if (saved) loadPaths(); else setPaths([]); }, [saved]);

  const mapQuery = mapPath ? `path=${encodeURIComponent(mapPath)}${corners.trim() ? `&corners=${encodeURIComponent(corners.trim())}` : ""}` : "";
  useEffect(() => {
    setInfo(null);
    if (!mapPath) return;
    api(`/map-info?${mapQuery}`).then(setInfo).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message }));
  }, [mapQuery]);

  // Fit as soon as there are four points (nothing is saved until Save).
  useEffect(() => {
    if (pts.length < 4) { setFit(null); return; }
    api(`/missions/${missionId}/recordings/${versionId}/geo`, { method: "PUT", body: { points: pts, check_only: true, dtm_path: dtmPath } })
      .then((r) => setFit(r.fit)).catch((e) => { setFit(null); setMsg({ kind: "err", text: (e as ApiError).message }); });
  }, [pts, dtmPath]);

  const save = async () => {
    try {
      const r = await api(`/missions/${missionId}/recordings/${versionId}/geo`, { method: "PUT",
        body: { points: pts, map_path: mapPath, dtm_path: dtmPath, corners: corners.trim() ? corners.split(",").map(Number) : null } });
      setSaved(r.geo);
      setMsg({ kind: "ok", text: "The camera is on the map. Every video from the same folder uses it." });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const clear = async () => {
    await api(`/missions/${missionId}/recordings/${versionId}/geo`, { method: "DELETE" });
    setSaved(null); setPts([]); setFit(null); setMsg({ kind: "ok", text: "The camera is no longer on the map." });
  };

  const centres: { label: string; ll: LatLon | null }[] = [
    { label: "The points picked", ll: pts.length ? [pts.reduce((a, p) => a + p[2], 0) / pts.length, pts.reduce((a, p) => a + p[3], 0) / pts.length] : null },
    { label: "Where the camera looks (its footprint)", ll: fp?.bounds ? [(fp.bounds[0] + fp.bounds[2]) / 2, (fp.bounds[1] + fp.bounds[3]) / 2] : null },
    { label: "The map shown now", ll: info ? [(info.top + info.bottom) / 2, (info.left + info.right) / 2] : null },
    { label: "The mission's place", ll: home },
  ];
  const onFrame = (x: number, y: number) => { setHalf([x, y]); setMsg(null); };
  const onMap = (lat: number, lon: number) => {
    if (boxMode === "start") { setBoxMode([lat, lon]); return; }
    if (boxMode) { savePlat({ ...plat, search_box: [boxMode[0], boxMode[1], lat, lon] }); setBoxMode(null); return; }
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
        <MapSource role="map" cat={cat} path={mapPath} onPath={(p) => { if (p !== mapPath) { setMapPath(p); setCorners(""); } }}
          centres={centres} onCatalogue={loadCat} />
        {mapPath && !mapPath.match(/\.tiff?$/i) && <label className="small">Corners: top latitude, left longitude, bottom latitude, right longitude
          <input value={corners} onChange={(e) => setCorners(e.target.value)} placeholder="32.0805,34.7795,32.0725,34.7875" /></label>}
        {info && <div className="muted small">{mapPath?.replace(/^.*[\\/]/, "")}: {info.width}×{info.height} pixels, {info.m_per_px} m a pixel, from {info.top.toFixed(5)}°, {info.left.toFixed(5)}° to {info.bottom.toFixed(5)}°, {info.right.toFixed(5)}°</div>}
      </div>

      <div className="card grid" style={{ gap: 8 }}>
        <strong className="small">Ground heights <span className="muted">(optional)</span></strong>
        <MapSource role="terrain" cat={cat} path={dtmPath} onPath={setDtmPath} centres={centres} onCatalogue={loadCat} />
        {dtm && <div className="muted small">{dtm.name}: heights {dtm.min_m} to {dtm.max_m} m, a height every {dtm.m_per_sample} m.
          {info && (dtm.top < info.bottom || dtm.bottom > info.top || dtm.right < info.left || dtm.left > info.right) ? " It does not cover the map's area." : ""}</div>}
      </div>

      {!saved && <div className="card grid" style={{ gap: 8 }}>
        <strong className="small">Drone or aircraft: platform data <span className="muted">(when the video has none, or it is wrong)</span></strong>
        <div className="muted small">{tel?.source ? `The video has flight data (${tel.source === "dji-srt" ? "DJI subtitle file" : "KLV, STANAG 4609"}). What you type here wins over it.`
          : "The video has no flight data. Type what you know, or draw a square on the map where it was flown, and Prepare will search there."}</div>
        <PlatformForm value={plat} onSave={savePlat} hasDtm={!!dtmPath} />
        <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <button disabled={!info} onClick={() => setBoxMode(boxMode ? null : "start")}>{boxMode ? "Cancel the square" : plat.search_box ? "Draw the search square again" : "Draw a search square on the map"}</button>
          {plat.search_box && <button onClick={() => savePlat({ ...plat, search_box: null })}>Remove the square</button>}
          <span className="muted small">{boxMode === "start" ? "Click one corner of the square on the map…" : boxMode ? "…and now the opposite corner." :
            plat.search_box ? "Without a position, the first frame is looked for only inside the square." : ""}</span>
        </div>
      </div>}

      {!saved && mapPath && info && <div className="card grid" style={{ gap: 8 }}>
        <strong className="small">Prepare the whole video on the map</strong>
        <div className="muted small">LanePulse lines up every frame with the map once, now, and keeps the result. Then you can play the video back
          and watch it move on the map in real time, with nothing to wait for.</div>
        <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <button className={prep.track ? "" : "primary"} disabled={prep.job?.state === "running"} onClick={prepare}>
            {prep.job?.state === "running" ? "Preparing…" : prep.track ? "Prepare again" : "Prepare"}</button>
          {prep.track && <button className="primary" onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "▶ Play on the map"}</button>}
          {prep.job?.state === "running" && <span className="small">{prep.job.done}{prep.job.total ? ` of ${prep.job.total}` : ""} moments</span>}
          {prep.job?.state === "failed" && <span className="msg err small">{prep.job.error}</span>}
          {prep.track && <span className="muted small">Lined up with the map: {pct(prep.track.summary["map-match"], prep.track)} ·
            flight data only: {pct(prep.track.summary["flight-data"], prep.track)} ·
            {(prep.track.summary.sky ?? 0) > 0 && <> sees the sky (paused): {pct(prep.track.summary.sky, prep.track)} · </>}not placed: {pct(prep.track.summary.lost, prep.track)} ·
            prepared in {prep.track.seconds} s</span>}
        </div>
        {prep.job?.state === "running" && prep.job.total && <div className="progress"><div style={{ width: `${Math.min(100, 100 * prep.job.done / prep.job.total)}%` }} /></div>}
      </div>}

      <div className="card grid" style={{ gap: 8 }}>
        <strong className="small">2. The same spot on the frame, then on the map, four times or more</strong>
        <div className="muted small">Pick spots on the ground that you can find on both: road marking corners, kerb corners, manhole covers.
          Spread them across the road, near and far. Six or more is better than four.</div>
        <div className="row" style={{ gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 420px", minWidth: 300 }}>
            {rec && (playing
              ? <PlayPanel src={`/api/missions/${missionId}/videos/${versionId}`} t={t} onTime={setT} onEnd={() => setPlaying(false)} />
              : <FramePanel missionId={missionId} versionId={versionId} rec={rec} t={t} pts={pts} half={half} fit={fit}
                paths={[]} onPick={onFrame} />)}
            {prep.track && <Timeline track={prep.track} t={t} duration={rec?.duration ?? 1} onSeek={(x) => { setT(x); }} />}
            {rec && <label className="small row" style={{ gap: 8 }}>Frame at {fmtT(t)}
              <input type="range" min={0} max={Math.max(0.1, rec.duration - 0.1)} step={0.1} value={t} onChange={(e) => setT(Number(e.target.value))} style={{ flex: 1 }} /></label>}
          </div>
          <div style={{ flex: "1 1 420px", minWidth: 300 }}>
            {info && mapPath ? <MapPanel src={`/api/map-image?${mapQuery}`} info={info} pts={pts} fit={fit} paths={paths} onPick={onMap} waiting={!!half || !!boxMode}
              fp={layers.footprint ? (playing ? trackFp(t) : fp) : null}
              box={plat.search_box ?? null} boxStart={Array.isArray(boxMode) ? boxMode : null}
              frameSrc={!playing && layers.frame && fp?.bounds ? `/api/missions/${missionId}/recordings/${versionId}/frame-on-map?t=${t.toFixed(1)}${dtmQ}&v=${saved?.created_at ?? ""}${fit ? fit.rms_m : ""}` : null}
              shadeSrc={layers.shade && dtmPath ? `/api/terrain-shade?path=${encodeURIComponent(dtmPath)}&top=${info.top}&left=${info.left}&bottom=${info.bottom}&right=${info.right}&w=${Math.min(2048, info.width)}` : null} />
              : <div className="muted small" style={{ padding: 24 }}>Pick or fetch a map first.</div>}
            {info && mapPath && <div className="row small maplayers" style={{ gap: 14, flexWrap: "wrap", marginTop: 6 }}>
              <label className="row" style={{ gap: 6 }}><input type="checkbox" checked={layers.frame} onChange={(e) => setLayers({ ...layers, frame: e.target.checked })} />The frame on the map</label>
              <label className="row" style={{ gap: 6 }}><input type="checkbox" checked={layers.footprint} onChange={(e) => setLayers({ ...layers, footprint: e.target.checked })} />Footprint</label>
              {dtmPath && <label className="row" style={{ gap: 6 }}><input type="checkbox" checked={layers.shade} onChange={(e) => setLayers({ ...layers, shade: e.target.checked })} />Hill shading</label>}
              <FootprintKey fp={playing ? trackFp(t) : fp} aligning={aligning && !playing} />
            </div>}
          </div>
        </div>
        {pts.length > 0 && <table className="small">
          <thead><tr><th>#</th><th>Frame</th><th>Map</th>{fit?.heights_m && <th>Ground height</th>}<th>Error</th><th>Left out</th><th /></tr></thead>
          <tbody>{pts.map((p, i) => <tr key={i} style={fit?.suspect === i ? { color: "var(--bad, #e84a6a)" } : undefined}>
            <td>{i + 1}</td><td className="mono">{p[0]}, {p[1]}</td><td className="mono">{p[2].toFixed(6)}, {p[3].toFixed(6)}</td>
            {fit?.heights_m && <td>{fit.heights_m[i] != null ? `${fit.heights_m[i]} m` : "–"}</td>}
            <td>{fit ? `${fit.residual_m[i]} m` : "–"}</td>
            <td>{fit?.leave_one_out_m[i] != null ? `${fit.leave_one_out_m[i]} m` : "–"}{fit?.suspect === i ? " · probably a wrong click" : ""}</td>
            <td><button onClick={() => setPts(pts.filter((_, k) => k !== i))} aria-label={`Remove point ${i + 1}`}>✕</button></td>
          </tr>)}</tbody>
        </table>}
        {half && <div className="muted small">Now click the same spot on the map.</div>}
        {fit && <div className="small">Fit: {fit.rms_m} m on average over the points
          {fit.leave_one_out_m.length > 0 && <>; a new point would be about {median(fit.leave_one_out_m)} m off</>}.</div>}
        {fit?.not_flat && <div className="msg err small">The points are {fit.height_span_m} m apart in height. The camera is placed as if the ground were flat,
          so expect errors of a few metres on the slope. Pick points at about the same height as the road you measure.</div>}
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
function MapPanel({ src, info, pts, fit, paths, onPick, waiting, fp, frameSrc, shadeSrc, box, boxStart }: {
  src: string; info: MapInfo; pts: Pt[]; fit: Fit | null; paths: MPath[]; onPick: (lat: number, lon: number) => void; waiting: boolean;
  fp: Footprint | null; frameSrc: string | null; shadeSrc: string | null; box: [number, number, number, number] | null; boxStart: LatLon | null;
}) {
  const W = info.width, H = info.height;
  const [vb, setVb] = useState({ x: 0, y: 0, w: W, h: H });
  useEffect(() => setVb({ x: 0, y: 0, w: W, h: H }), [W, H, src]);
  // Start on the frame's footprint, the points already picked, or the targets.
  const hasFp = !!fp?.bounds;
  // the whole view, but not farther out than half the map's size again around the map
  const clip = (b: [number, number, number, number]) => {
    const dl = (info.top - info.bottom) / 2, dw = (info.right - info.left) / 2;
    const c = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
    return [[c(b[0], info.bottom - dl, info.top + dl), c(b[1], info.left - dw, info.right + dw)],
      [c(b[2], info.bottom - dl, info.top + dl), c(b[3], info.left - dw, info.right + dw)]];
  };
  useEffect(() => {
    const ll = [...pts.map((p) => [p[2], p[3]]), ...paths.flatMap((p) => p.path.map((q) => [q[1], q[2]])),
      ...(fp?.view_bounds ? clip(fp.view_bounds) : fp?.bounds ? [[fp.bounds[0], fp.bounds[1]], [fp.bounds[2], fp.bounds[3]]] : [])];
    if (!ll.length) return;
    const xs = ll.map((q) => toX(q[1])), ys = ll.map((q) => toY(q[0]));
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const side = Math.max(x1 - x0, y1 - y0, 80) * 1.6;
    setVb({ x: (x0 + x1) / 2 - side / 2, y: (y0 + y1) / 2 - side / 2, w: side, h: side });
  }, [paths.length > 0, info, hasFp]);
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
        {shadeSrc && <image href={shadeSrc} width={W} height={H} preserveAspectRatio="none" opacity={0.75} style={{ mixBlendMode: "hard-light" }} />}
        {frameSrc && fp?.bounds && <image href={frameSrc} x={toX(fp.bounds[1])} y={toY(fp.bounds[0])} preserveAspectRatio="none"
          width={toX(fp.bounds[3]) - toX(fp.bounds[1])} height={toY(fp.bounds[2]) - toY(fp.bounds[0])} opacity={0.85} />}
        {fp?.far && <polygon points={fp.far.map((q) => `${toX(q[1]).toFixed(1)},${toY(q[0]).toFixed(1)}`).join(" ")}
          fill={TOP} fillOpacity={0.14} stroke="none"><title>The upper part of the image (toward the horizon)</title></polygon>}
        {fp?.near && fp.far && <polygon points={[fp.far[0], fp.far[1], fp.near[2], fp.near[3]].map((q) => `${toX(q[1]).toFixed(1)},${toY(q[0]).toFixed(1)}`).join(" ")}
          fill="none" stroke={fpColour(fp)} strokeWidth={r / 2.5} strokeLinejoin="round"><title>{fpTitle(fp)}</title></polygon>}
        {fp?.far && <line x1={toX(fp.far[0][1])} y1={toY(fp.far[0][0])} x2={toX(fp.far[1][1])} y2={toY(fp.far[1][0])}
          stroke={TOP} strokeWidth={r / 1.2} strokeLinecap="round">
          <title>{fp.horizon ? (fp.beyond ? "The far edge of the view (5 km): the camera sees on toward the horizon" : "The horizon: above it the camera sees sky")
            : "The top edge of the image"}</title></line>}
        {lines.map((l, i) => <path key={i} d={l.d} fill="none" stroke={colour(l.cls)} strokeWidth={r / 2.5} strokeOpacity={0.85}>
          <title>{l.key} · {l.cls}</title></path>)}
        {box && <rect x={Math.min(toX(box[1]), toX(box[3]))} y={Math.min(toY(box[0]), toY(box[2]))} width={Math.abs(toX(box[3]) - toX(box[1]))}
          height={Math.abs(toY(box[2]) - toY(box[0]))} fill="none" stroke="#b48cff" strokeWidth={r / 3} strokeDasharray={`${r * 2} ${r}`}>
          <title>The search square: where the video is looked for when it has no position</title></rect>}
        {boxStart && <circle cx={toX(boxStart[1])} cy={toY(boxStart[0])} r={r} fill="#b48cff" />}
        {pts.map((p, i) => <Mark key={i} x={toX(p[3])} y={toY(p[2])} r={r} n={i + 1} bad={fit?.suspect === i} />)}
      </svg>
      <div className="muted small">Wheel to zoom, drag to move.</div>
    </div>
  );
}

const GREEN = "#3fe08a", RED = "#ff4d5e", TOP = "#ffd23f", SKY = "#5aa9ff";
const SKY_NOTE = "The camera sees the sky here, so it is near level: geo-registration waits until it sees only ground.";
const fpColour = (fp: Footprint) => (fp.registered ? GREEN : RED);
const fpTitle = (fp: Footprint) => fp.how === "points" ? "Registered: placed by the points on the map"
  : fp.how === "map-match" ? "Registered: lined up with the map by matching"
  : fp.how === "points-poor" ? "Not registered well: the points disagree by more than 3 m" : "Not registered: placed by the flight data only";

function FootprintKey({ fp, aligning }: { fp: Footprint | null; aligning: boolean }) {
  if (aligning) return <span className="muted">Lining the frame up with the map…</span>;
  if (!fp) return <span className="muted">Not on the map yet: pick four points, or a recording with flight data.</span>;
  if (fp.how === "sky") return <span><span style={{ color: SKY }}>■</span> <span className="muted">{fp.note || SKY_NOTE}</span></span>;
  return <span className="row" style={{ gap: 10, flexWrap: "wrap" }}>
    <span><span style={{ color: fpColour(fp) }}>▭</span> Outline: {fpTitle(fp)}</span>
    <span><span style={{ color: TOP }}>━</span> {fp.horizon ? (fp.beyond ? "far edge of the view, toward the horizon" : "the horizon: above it the camera sees sky") : "top edge of the image"}</span>
    <span><span style={{ color: TOP, opacity: 0.6 }}>■</span> upper part of the image</span>
    {fp.note && <span className="muted">{fp.note}</span>}
  </span>;
}

const PITCH: [number, string][] = [[-90, "Straight down (-90°)"], [-75, "Steep (-75°)"], [-60, "-60°"], [-45, "-45°"], [-30, "-30°"], [-15, "Shallow (-15°)"]];
const FOV: [number, string][] = [[84, "84° (DJI wide camera)"], [73.7, "74° (24 mm equivalent)"], [60, "60°"], [45, "45°"], [30, "30° (zoomed in)"], [15, "15° (long zoom)"], [5, "5° (very long zoom)"]];

/** Platform data the operator knows: position, height (above sea level or the ground), heading, camera angle, lens. */
function PlatformForm({ value, onSave, hasDtm }: { value: Platform; onSave: (p: Platform) => void; hasDtm: boolean }) {
  const [pos, setPos] = useState("");
  const [hKind, setHKind] = useState<"asl" | "agl">("agl");
  const [hv, setHv] = useState("");
  const [heading, setHeading] = useState("");
  const [pitch, setPitch] = useState<number | null>(null);
  const [fov, setFov] = useState<number | null>(null);
  useEffect(() => {
    setPos(value.lat != null && value.lon != null ? `${value.lat}, ${value.lon}` : "");
    setHKind(value.alt_asl != null ? "asl" : "agl");
    setHv(value.alt_asl != null ? String(value.alt_asl) : value.agl != null ? String(value.agl) : "");
    setHeading(value.heading != null ? String(value.heading) : "");
    setPitch(value.pitch ?? null); setFov(value.hfov ?? null);
  }, [value]);
  const save = () => {
    const m = pos.match(/(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)/);
    const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
    onSave({ search_box: value.search_box ?? null, lat: m ? Number(m[1]) : undefined, lon: m ? Number(m[2]) : undefined,
      alt_asl: hKind === "asl" ? num(hv) : undefined, agl: hKind === "agl" ? num(hv) : undefined,
      heading: num(heading), pitch: pitch ?? undefined, hfov: fov ?? undefined });
  };
  return <div className="grid platform" style={{ gap: 8 }}>
    <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
      <label className="small grow">Position (latitude, longitude)<input value={pos} onChange={(e) => setPos(e.target.value)} placeholder="32.7905, 35.0090" /></label>
      <label className="small">Height<Choice label="Height measured from" value={hKind} options={[["agl", "Above the ground"], ["asl", "Above sea level"]]} onChange={(v) => setHKind(v as "asl" | "agl")} /></label>
      <label className="small" style={{ width: 120 }}>Metres<input type="number" value={hv} onChange={(e) => setHv(e.target.value)} placeholder="120" /></label>
    </div>
    <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
      <label className="small" style={{ width: 150 }}>Heading (° from north)<input type="number" value={heading} onChange={(e) => setHeading(e.target.value)} placeholder="0" /></label>
      <label className="small grow">Camera angle<Choice label="Camera angle" value={pitch ?? ""} placeholder="From the video, or straight down" options={PITCH} onChange={setPitch} /></label>
      <label className="small grow">Field of view<Choice label="Field of view" value={fov ?? ""} placeholder="From the video" options={FOV} onChange={setFov} /></label>
      <button style={{ alignSelf: "end" }} onClick={save}>Save platform data</button>
    </div>
    {hKind === "asl" && hv && <div className="muted small">{hasDtm ? "The ground's height under the platform comes from the ground heights above, so LanePulse knows how high above the ground it is."
      : "Add ground heights above: without them the ground is taken at sea level, and the height above the ground will be wrong on high ground."}</div>}
  </div>;
}

/** The recording playing; its time drives the outline on the map. */
function PlayPanel({ src, t, onTime, onEnd }: { src: string; t: number; onTime: (t: number) => void; onEnd: () => void }) {
  const v = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const el = v.current;
    if (!el) return;
    el.currentTime = t;
    el.play().catch(() => onEnd());
    let id = 0;
    const tick = () => { onTime(el.currentTime); id = requestAnimationFrame(tick); };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, []);
  return <video ref={v} src={src} muted playsInline onEnded={onEnd} style={{ width: "100%", background: "#000", display: "block" }} />;
}

/** The prepared video along time: green where lined up with the map, red on flight data only, blue where the camera
 * sees the sky (paused), grey where not placed. */
function Timeline({ track, t, duration, onSeek }: { track: Track; t: number; duration: number; onSeek: (t: number) => void }) {
  const W = 1000;
  const step = 1 / track.fps;
  return <svg viewBox={`0 0 ${W} 14`} preserveAspectRatio="none" style={{ width: "100%", height: 14, cursor: "pointer", display: "block", marginTop: 4 }}
    onClick={(e) => { const b = (e.currentTarget as SVGSVGElement).getBoundingClientRect(); onSeek((e.clientX - b.left) / b.width * duration); }}
    role="img" aria-label="Prepared moments: green lined up with the map, red flight data only, blue the camera sees the sky, grey not placed">
    {track.outlines.map((o, i) => <rect key={i} x={o.t / duration * W} width={step / duration * W + 0.5} y={2} height={10}
      fill={o.how === "map-match" ? GREEN : o.how === "flight-data" ? RED : o.how === "sky" ? SKY : "#556070"} />)}
    <rect x={t / duration * W - 1.5} width={3} y={0} height={14} fill="#fff" />
  </svg>;
}
