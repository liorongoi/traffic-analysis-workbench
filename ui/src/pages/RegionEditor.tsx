import { useEffect, useRef, useState } from "react";
import { api, ApiError, post, Region } from "../api";

// Points are stored in the view's original pixels; display size and zoom never change them (VIEW-03).
export default function RegionEditor({ missionId, viewId }: { missionId: string; viewId: number }) {
  const [data, setData] = useState<{ width: number; height: number; representatives: Record<string, number[]>; regions: Region[] } | null>(null);
  const [bgId, setBgId] = useState<number | null>(null);
  const [kind, setKind] = useState<"include" | "exclude">("include");
  const [name, setName] = useState("");
  const [pts, setPts] = useState<[number, number][]>([]);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const load = async () => {
    const d = await api(`/missions/${missionId}/views/${viewId}/regions`);
    setData(d);
    if (bgId == null) setBgId((d.representatives.colour ?? Object.values(d.representatives)[0] ?? [null])[0]);
  };
  useEffect(() => { load(); }, [missionId, viewId]);

  const toImage = (e: React.MouseEvent): [number, number] => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [Math.round(Math.max(0, Math.min(data!.width, p.x))), Math.round(Math.max(0, Math.min(data!.height, p.y)))];
  };
  const save = async () => {
    setErr(null);
    try {
      await post(`/missions/${missionId}/views/${viewId}/regions`, { kind, polygon: pts, name: name || null });
      setPts([]);
      setName("");
      load();
    } catch (e) { setErr((e as ApiError).message); }
  };
  const del = async (uid: string) => { await api(`/missions/${missionId}/views/${viewId}/regions/${uid}`, { method: "DELETE" }); load(); };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "Escape") setPts([]);
      if (e.key === "Backspace") setPts((p) => p.slice(0, -1));
      if (e.key === "Enter" && pts.length >= 3) save();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!data) return <div className="page muted">Loading…</div>;
  const sw = Math.max(1.5, data.width / 640);
  return (
    <div className="page" style={{ maxWidth: "none" }}>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <a href={`#/m/${missionId}/views`}>← Camera views</a>
        <div className="row">
          <span className="muted">Background image:</span>
          {Object.entries(data.representatives).map(([c, ids]) => ids.map((id) => (
            <button key={id} className={bgId === id ? "primary" : ""} onClick={() => setBgId(id)}>{c}</button>
          )))}
        </div>
      </div>
      <div className="inv">
        <div className="stage">
          {bgId && <img src={`/api/missions/${missionId}/images/${bgId}`} alt="" draggable={false} />}
          <svg ref={svgRef} viewBox={`0 0 ${data.width} ${data.height}`} preserveAspectRatio="xMidYMid meet" style={{ cursor: "crosshair" }}
            onClick={(e) => setPts([...pts, toImage(e)])} onMouseMove={(e) => setHover(toImage(e))} onMouseLeave={() => setHover(null)}>
            <defs>
              <pattern id="hatch2" width={10 * sw} height={10 * sw} patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
                <line x1="0" y1="0" x2="0" y2={10 * sw} stroke="var(--exclude)" strokeWidth={sw} />
              </pattern>
            </defs>
            {data.regions.map((r) => (
              <g key={r.uid}>
                <polygon points={r.polygon.map((p) => p.join(",")).join(" ")} fill={r.kind === "exclude" ? "url(#hatch2)" : "rgba(90,150,255,0.18)"}
                  stroke={r.kind === "exclude" ? "var(--exclude)" : "var(--include)"} strokeWidth={sw * 1.5} strokeDasharray={r.kind === "exclude" ? `${6 * sw} ${3 * sw}` : undefined} />
                <text x={r.polygon[0][0]} y={r.polygon[0][1] - 4 * sw} fill="#fff" fontSize={12 * sw} paintOrder="stroke" stroke="#000" strokeWidth={3 * sw}>
                  {r.kind === "exclude" ? "Excluded" : "Included"}{r.name ? `: ${r.name}` : ""}</text>
              </g>
            ))}
            {pts.length > 0 && (
              <polyline points={[...pts, ...(hover ? [hover] : [])].map((p) => p.join(",")).join(" ")} fill="rgba(255,255,255,.12)"
                stroke={kind === "exclude" ? "var(--exclude)" : "var(--include)"} strokeWidth={sw * 2} />
            )}
            {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={4 * sw} fill="#fff" stroke="#000" strokeWidth={sw} />)}
          </svg>
        </div>
        <aside className="side">
          <div className="card">
            <h2>New region</h2>
            <p className="muted">Click on the image to add corners. <kbd>Backspace</kbd> removes the last one, <kbd>Enter</kbd> saves, <kbd>Esc</kbd> clears.</p>
            <div className="row">
              <label className="row"><input type="radio" checked={kind === "include"} onChange={() => setKind("include")} /> Include (road to analyse)</label>
              <label className="row"><input type="radio" checked={kind === "exclude"} onChange={() => setKind("exclude")} /> Exclude</label>
            </div>
            <label className="field" style={{ marginTop: 8 }}><span>Name (optional)</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. parking lot" /></label>
            <div className="mono muted">{pts.length} corner(s){hover ? ` · pointer at ${hover[0]}, ${hover[1]} px` : ""}</div>
            {err && <div className="msg err">{err}</div>}
            <div className="row" style={{ marginTop: 8 }}>
              <button className="primary" disabled={pts.length < 3} onClick={save}>Save region</button>
              <button onClick={() => setPts([])}>Clear</button>
            </div>
          </div>
          <div className="card">
            <h2>Regions of this view</h2>
            <p className="muted">Exclusion wins over inclusion. With no inclusion region the whole frame is analysed. A vehicle counts when the middle of its bottom edge is inside.</p>
            {data.regions.length === 0 && <p className="muted">None yet.</p>}
            {data.regions.map((r) => (
              <div key={r.uid} className="row" style={{ justifyContent: "space-between", padding: "4px 0" }}>
                <span><span className="badge">{r.kind}</span> {r.name ?? <span className="muted">unnamed</span>} <span className="muted">({r.polygon.length} corners)</span></span>
                <button className="danger" onClick={() => del(r.uid)}>Delete</button>
              </div>
            ))}
            <p className="muted">Edits are kept as history. Night candidates follow new regions at once; detections already made keep the regions they were processed with until reprocessed.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
