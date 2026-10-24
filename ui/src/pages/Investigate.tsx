import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, Box, Candidate, Item, Obj, post, REVIEW_LABEL } from "../api";
import { go } from "../router";
import { loadQuery } from "./Mission";

const CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"];
type Sel = { kind: "obj" | "cand"; uid: string } | null;

// Machine, human and candidate marks differ by line style and label, not by colour alone (UX-01).
const STYLE = {
  machine: { stroke: "var(--machine)", dash: "" },
  human: { stroke: "var(--human)", dash: "8 4" },
  candidate: { stroke: "var(--candidate)", dash: "2 4" },
  confirmed: { stroke: "var(--human)", dash: "10 3 2 3" },
  plate: { stroke: "var(--plate)", dash: "" },
  plateHuman: { stroke: "var(--plate)", dash: "6 3" },
};
const PLATE_REVIEW: Record<string, string> = {
  pending: "not reviewed", confirmed: "confirmed", corrected: "corrected", unreadable: "unreadable (operator)", no_plate: "no plate (operator)",
};

export default function Investigate({ missionId, versionId }: { missionId: string; versionId: number }) {
  const [item, setItem] = useState<Item | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<Sel>(null);
  const [overlays, setOverlays] = useState(true);
  const [showRemoved, setShowRemoved] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [drawCls, setDrawCls] = useState("car");
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [pickedLights, setPickedLights] = useState<number[]>([]);
  const [help, setHelp] = useState(false);
  const [plates, setPlates] = useState<Record<string, any>>({});
  const [drawPlate, setDrawPlate] = useState(false);
  const [plateBox, setPlateBox] = useState<number[] | null>(null);
  const [plateText, setPlateText] = useState("");
  const svgRef = useRef<SVGSVGElement>(null);
  const q = loadQuery(missionId);

  const load = useCallback(async () => {
    try {
      setItem(await post(`/missions/${missionId}/items/${versionId}?sort=${q.sort}&desc=${q.desc}`, q.filter));
      setErr(null);
    } catch (e) { setErr((e as ApiError).message); }
  }, [missionId, versionId]);
  const loadPlates = useCallback(async () => {
    try {
      const r = await api(`/missions/${missionId}/plates?version_id=${versionId}&limit=500`);
      setPlates(Object.fromEntries(r.plates.map((p: any) => [p.object_uid, p])));
    } catch { setPlates({}); }        // not a plate recognition mission
  }, [missionId, versionId]);
  useEffect(() => { setSel(null); setPickedLights([]); load(); loadPlates(); }, [load, loadPlates]);
  useEffect(() => {
    setDrawPlate(false); setPlateBox(null); setPlateText("");
    // Bring the selected object's panel into view when it was picked from a long list or the image.
    if (sel) requestAnimationFrame(() => document.querySelector(".side .selcard")?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  }, [sel?.uid]);

  const run = async (fn: () => Promise<unknown>) => {
    try { await fn(); await load(); } catch (e) { setErr((e as ApiError).message); }
  };
  const base = `/missions/${missionId}/items/${versionId}`;
  const objAct = (uid: string, body: object) => run(() => api(`${base}/objects/${uid}`, { method: "PATCH", body }));
  const candAct = (uid: string, body: object) => run(() => post(`${base}/candidates/${uid}`, body));
  const approve = () => run(() => post(`${base}/decision/approve`));
  const plateAct = (pid: number, body: object) => run(async () => { await post(`/missions/${missionId}/plates/${pid}/revise`, body); await loadPlates(); });

  const selObj = item?.objects.find((o) => sel?.kind === "obj" && o.uid === sel.uid);
  const selCand = item?.candidates.find((c) => sel?.kind === "cand" && c.uid === sel.uid);
  const selPlate = selObj ? plates[selObj.uid] : undefined;
  const drawAny = drawing || drawPlate;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!item || (e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
      const k = e.key.toLowerCase();
      if (k === "j" && item.position.next) go(`/m/${missionId}/i/${item.position.next}`);
      else if (k === "k" && item.position.prev) go(`/m/${missionId}/i/${item.position.prev}`);
      else if (k === "a") approve();
      else if (k === "b") setDrawing((d) => !d);
      else if (k === "h") setOverlays((o) => !o);
      else if (k === "escape") { setDrawing(false); setDrawPlate(false); setPlateBox(null); setSel(null); setPickedLights([]); setHelp(false); }
      else if (k === "?") setHelp((h) => !h);
      else if ((k === "delete" || k === "backspace") && selObj) objAct(selObj.uid, { action: selObj.status === "removed" ? "restore" : "remove" });
      else if ((k === "delete" || k === "backspace") && selCand) candAct(selCand.uid, { action: "reject" });
      else if (k === "c" && selCand) candAct(selCand.uid, { action: "confirm", cls: drawCls });
      else if (k === "g" && pickedLights.length) run(() => post(`${base}/regroup`, { light_ids: pickedLights })).then(() => setPickedLights([]));
      else if (/^[1-5]$/.test(k)) {
        const cls = CLASSES[Number(k) - 1];
        setDrawCls(cls);
        if (selObj && selObj.cls !== cls) objAct(selObj.uid, { action: "reclass", cls });
      } else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const toImage = (e: React.MouseEvent) => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return { x: Math.max(0, Math.min(item!.width!, p.x)), y: Math.max(0, Math.min(item!.height!, p.y)) };
  };

  if (err && !item) return <div className="page"><div className="msg err">{err}</div></div>;
  if (!item) return <div className="page muted">Loading…</div>;
  const W = item.width ?? 1280, H = item.height ?? 720;
  const sw = Math.max(1.5, W / 640);
  const counted = item.objects.filter((o) => o.counted).length + item.candidates.filter((c) => c.status === "confirmed").length;
  const open = item.candidates.filter((c) => c.status === "unresolved");
  const cmin = open.reduce((a, c) => a + c.count_min, 0), cmax = open.reduce((a, c) => a + c.count_max, 0);

  const rect = (b: Box, s: { stroke: string; dash: string }, label: string, selected: boolean, onClick: () => void, faint = false) => (
    <g onClick={(e) => { e.stopPropagation(); if (!drawAny) onClick(); }} style={{ cursor: drawAny ? "crosshair" : "pointer" }} opacity={faint ? 0.45 : 1}>
      <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="transparent" stroke={s.stroke}
        strokeWidth={selected ? sw * 2.2 : sw} strokeDasharray={s.dash ? s.dash.split(" ").map((n) => Number(n) * sw).join(" ") : undefined} />
      <text x={b[0]} y={b[1] - 3 * sw} fill={s.stroke} fontSize={11 * sw} paintOrder="stroke" stroke="#000" strokeWidth={2.5 * sw}>{label}</text>
    </g>
  );

  return (
    <div className="page" style={{ maxWidth: "none" }}>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <div className="row">
          <a href={`#/m/${missionId}/results`}>← Results</a>
          <button disabled={!item.position.prev} onClick={() => go(`/m/${missionId}/i/${item.position.prev}`)}>‹ Previous <kbd>K</kbd></button>
          <span><b>{item.position.index ?? "–"}</b> of {item.position.total}</span>
          <button disabled={!item.position.next} onClick={() => go(`/m/${missionId}/i/${item.position.next}`)}>Next <kbd>J</kbd> ›</button>
        </div>
        <div className="row">
          <button className={drawing ? "primary" : ""} onClick={() => setDrawing(!drawing)}>{drawing ? "Drawing: drag a box" : "Draw missed vehicle"} <kbd>B</kbd></button>
          <select value={drawCls} onChange={(e) => setDrawCls(e.target.value)}>{CLASSES.map((c, i) => <option key={c} value={c}>{i + 1} {c}</option>)}</select>
          <button onClick={() => setOverlays(!overlays)}>{overlays ? "Hide" : "Show"} overlays <kbd>H</kbd></button>
          <button className="primary" onClick={approve} disabled={item.processing_state !== "done"}
            title="Approve means every vehicle inside the included regions is marked correctly. The image can then train and test models, and unmarked areas count as road with no vehicle.">Approve <kbd>A</kbd></button>
          <button onClick={() => setHelp(true)}><kbd>?</kbd></button>
        </div>
      </div>
      {err && <div className="msg err">{err}</div>}
      <div className="inv">
        <div className="stage">
          <img src={`/api/missions/${missionId}/images/${versionId}`} alt={item.file} draggable={false} />
          <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
            style={{ cursor: drawAny ? "crosshair" : "default" }}
            onMouseDown={(e) => { if (drawAny) { const p = toImage(e); setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }); } }}
            onMouseMove={(e) => { if (drag) { const p = toImage(e); setDrag({ ...drag, x1: p.x, y1: p.y }); } }}
            onMouseUp={() => {
              if (drag) {
                const box = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)];
                setDrag(null);
                if (drawPlate) { if (box[2] - box[0] > 3 && box[3] - box[1] > 1) { setPlateBox(box.map((v) => Math.round(v * 10) / 10)); setDrawPlate(false); } }
                else if (box[2] - box[0] > 3 && box[3] - box[1] > 3) run(() => post(`${base}/objects`, { box, cls: drawCls }));
              }
            }}
            onClick={() => !drawAny && setSel(null)}>
            <defs>
              <pattern id="hatch" width={10 * sw} height={10 * sw} patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
                <line x1="0" y1="0" x2="0" y2={10 * sw} stroke="var(--exclude)" strokeWidth={sw} />
              </pattern>
            </defs>
            {overlays && item.regions.map((r) => (
              <polygon key={r.uid} points={r.polygon.map((p) => p.join(",")).join(" ")}
                fill={r.kind === "exclude" ? "url(#hatch)" : "rgba(90,150,255,0.10)"}
                stroke={r.kind === "exclude" ? "var(--exclude)" : "var(--include)"} strokeWidth={sw} pointerEvents="none" />
            ))}
            {overlays && item.objects.filter((o) => o.status === "present" || showRemoved).map((o: Obj) => rect(
              o.box, o.origin === "machine" ? STYLE.machine : STYLE.human,
              `${o.origin === "machine" ? "M" : "H"} ${o.cls}${o.score != null && o.correction === "none" ? " " + o.score.toFixed(2) : ""}${o.status === "removed" ? " removed" : ""}${o.in_region === 0 ? " (outside region)" : ""}`,
              sel?.kind === "obj" && sel.uid === o.uid, () => setSel({ kind: "obj", uid: o.uid }), o.status === "removed" || o.in_region === 0))}
            {overlays && item.candidates.filter((c) => c.status === "unresolved" || c.status === "confirmed").map((c: Candidate) => rect(
              c.box, c.status === "confirmed" ? STYLE.confirmed : STYLE.candidate,
              c.status === "confirmed" ? `C ${c.cls ?? "car"}` : `? ${c.count_min === c.count_max ? c.count_min : `${c.count_min}–${c.count_max}`}`,
              sel?.kind === "cand" && sel.uid === c.uid, () => setSel({ kind: "cand", uid: c.uid })))}
            {overlays && Object.values(plates).filter((p: any) => (p.operator_box || p.box) && p.review !== "no_plate").map((p: any) => {
              const b = p.operator_box ?? p.box;
              const label = p.readability === "unreadable" ? "plate unreadable" : p.text ?? p.machine_display ?? "";
              return <g key={`pl${p.id}`} pointerEvents="none">
                <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="transparent" stroke={STYLE.plate.stroke} strokeWidth={sw}
                  strokeDasharray={p.operator_box ? `${6 * sw} ${3 * sw}` : undefined} />
                <text x={b[0]} y={b[3] + 11 * sw} fill={STYLE.plate.stroke} fontSize={11 * sw} paintOrder="stroke" stroke="#000" strokeWidth={2.5 * sw} direction="ltr" unicodeBidi="plaintext">{label}</text>
              </g>;
            })}
            {plateBox && <rect x={plateBox[0]} y={plateBox[1]} width={plateBox[2] - plateBox[0]} height={plateBox[3] - plateBox[1]} fill="rgba(255,216,77,.15)"
              stroke={STYLE.plate.stroke} strokeWidth={sw * 1.5} strokeDasharray={`${6 * sw} ${3 * sw}`} pointerEvents="none" />}
            {overlays && selCand && item.lights.map((l) => (
              <circle key={l.id} cx={l.cx} cy={l.cy} r={Math.max(4 * sw, Math.sqrt(l.area))} fill="none"
                stroke={pickedLights.includes(l.id) ? "var(--human)" : selCand.member_ids.includes(l.id) ? "var(--candidate)" : "rgba(255,255,255,.5)"}
                strokeWidth={sw} style={{ cursor: "pointer" }}
                onClick={(e) => { e.stopPropagation(); setPickedLights((p) => p.includes(l.id) ? p.filter((x) => x !== l.id) : [...p, l.id]); }} />
            ))}
            {drag && <rect x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)} width={Math.abs(drag.x1 - drag.x0)} height={Math.abs(drag.y1 - drag.y0)}
              fill={drawPlate ? "rgba(255,216,77,.15)" : "rgba(90,208,138,.15)"} stroke={drawPlate ? "var(--plate)" : "var(--human)"} strokeWidth={sw} strokeDasharray={`${8 * sw} ${4 * sw}`} />}
          </svg>
        </div>
        <aside className="side">
          <div className="card">
            <div className="mono bidi" style={{ wordBreak: "break-all" }}>{item.file}</div>
            <div className="muted" style={{ marginTop: 4 }}>
              UTC <span className="mono">{item.capture_utc}</span> ({item.capture_precision} precision)<br />
              Local <span className="mono">{item.local?.time.replace("T", " ")}</span> {item.local?.abbreviation} {item.local?.dst ? "(summer time)" : ""}<br />
              {item.zone} · view <b>{item.view.name ?? "none"}</b>{item.view.regions_applied === false ? " (regions not applied)" : ""} · {item.condition?.condition ?? ""}
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <span className={`badge ${item.review_status}`}>{REVIEW_LABEL[item.review_status]}</span>
              <span className="badge">{item.processing_state}</span>
            </div>
            {item.error && <div className="msg err">{item.error}</div>}
          </div>
          <div className="card grid cols-2" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <div className="stat"><div className="n">{counted}</div><div className="l">vehicles in this image</div></div>
            <div className="stat"><div className="n">{cmax ? `${cmin}–${cmax}` : 0}</div><div className="l">unresolved night candidates (not counted)</div></div>
          </div>
          {selObj && (
            <div className="card selcard">
              <h3>Selected: {selObj.origin === "machine" ? "detected" : "added by operator"} {selObj.cls}</h3>
              {selObj.machine_cls && selObj.machine_cls !== selObj.cls && <div className="muted">Detector said {selObj.machine_cls}</div>}
              <div className="row" style={{ marginTop: 6 }}>
                <select value={selObj.cls} onChange={(e) => objAct(selObj.uid, { action: "reclass", cls: e.target.value })}>{CLASSES.map((c) => <option key={c}>{c}</option>)}</select>
                <button className="danger" onClick={() => objAct(selObj.uid, { action: selObj.status === "removed" ? "restore" : "remove" })}>
                  {selObj.status === "removed" ? "Restore" : "Not a vehicle"} <kbd>Del</kbd></button>
              </div>
            </div>
          )}
          {selObj && selPlate && (
            <div className="card">
              <h3>Plate</h3>
              {(selPlate.operator_box || selPlate.box) && <img className="pl" style={{ maxWidth: "100%" }} alt="plate"
                src={selPlate.operator_box ? `/api/missions/${missionId}/crops/${versionId}?box=${selPlate.operator_box.join(",")}&h=56` : `/api/missions/${missionId}/plates/${selPlate.id}/crop?h=56`} />}
              <div className="row" style={{ marginTop: 4 }}>
                <span className="platetext"><bdi>{selPlate.readability === "unreadable" ? "unreadable" : selPlate.text ?? "–"}</bdi></span>
                {selPlate.readability && <span className="badge">{selPlate.readability}</span>}
                <span className="badge">{PLATE_REVIEW[selPlate.review]}</span>
                {selPlate.evaluation_only && <span className="badge flag">evaluation only</span>}
              </div>
              <div className="muted" style={{ fontSize: 12.5 }}>
                {selPlate.outcome === "read" ? <>Machine: <bdi className="mono">{selPlate.machine_display ?? "–"}</bdi> ({selPlate.machine_readability})</> :
                  selPlate.outcome === "no_plate_found" ? "The plate finder found no plate on this vehicle." :
                  selPlate.outcome === "vehicle_too_small" ? "The vehicle is too small; no plate was looked for." : "The plate is too small to read."}
                {selPlate.operator_box && " Box drawn by an operator."}
              </div>
              {selPlate.reasons.length > 0 && <ul className="reasons" style={{ margin: "4px 0" }}>{selPlate.reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul>}
              <div className="row" style={{ gap: 4, marginTop: 6 }}>
                <button disabled={selPlate.machine_readability !== "full"} onClick={() => plateAct(selPlate.id, { action: "confirm" })}>Confirm</button>
                <button onClick={() => plateAct(selPlate.id, { action: "unreadable" })}>Unreadable</button>
                <button onClick={() => plateAct(selPlate.id, { action: "no_plate" })}>No plate</button>
                {selPlate.review !== "pending" && <button onClick={() => plateAct(selPlate.id, { action: "reopen" })}>Reopen</button>}
              </div>
              <div className="row" style={{ gap: 4, marginTop: 6 }}>
                <button className={drawPlate ? "primary" : ""} onClick={() => { setDrawing(false); setDrawPlate(!drawPlate); }}>
                  {drawPlate ? "Drag around the plate…" : plateBox ? "Redraw plate box" : "Draw plate box"}</button>
                {plateBox && <button onClick={() => setPlateBox(null)}>Clear box</button>}
              </div>
              <div className="row" style={{ gap: 4, marginTop: 6 }}>
                <input dir="auto" placeholder="Type the plate as read" value={plateText} onChange={(e) => setPlateText(e.target.value)} style={{ width: 170 }} />
                <button className="primary" disabled={!plateText.trim()} onClick={() => {
                  plateAct(selPlate.id, { action: "correct", text: plateText, ...(plateBox ? { box: plateBox } : {}) });
                  setPlateText(""); setPlateBox(null);
                }}>Save{plateBox ? " with box" : ""}</button>
              </div>
              <span className="muted" style={{ fontSize: 12 }}>Type only what you can read. If a character is unclear, mark the plate unreadable.</span>
            </div>
          )}
          {selCand && (
            <div className="card selcard">
              <h3>Night candidate: {selCand.kind} of lights</h3>
              <div className="muted">Counts as {selCand.count_min}–{selCand.count_max}. {selCand.alternatives.join("; ")}</div>
              {selCand.flags.length > 0 && <div className="row" style={{ marginTop: 4 }}>{selCand.flags.map((f) => <span key={f} className="badge flag">{f.replace(/_/g, " ")}</span>)}</div>}
              <div className="row" style={{ marginTop: 8 }}>
                <button className="primary" onClick={() => candAct(selCand.uid, { action: "confirm", cls: drawCls })}>Vehicle ({drawCls}) <kbd>C</kbd></button>
                <button onClick={() => candAct(selCand.uid, { action: "reject" })}>Not a vehicle <kbd>Del</kbd></button>
                <button onClick={() => candAct(selCand.uid, { action: "persistent" })}>Fixed light (lamp, sign)</button>
              </div>
              <p className="muted">To regroup, click lights on the image (circles), then press <kbd>G</kbd>. {pickedLights.length > 0 && `${pickedLights.length} chosen.`}</p>
            </div>
          )}
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}><h3>Objects</h3>
              <label className="row muted"><input type="checkbox" checked={showRemoved} onChange={(e) => setShowRemoved(e.target.checked)} /> show removed</label></div>
            {item.objects.filter((o) => o.status === "present" || showRemoved).map((o) => (
              <div key={o.uid} className={`objrow ${sel?.uid === o.uid ? "sel" : ""}`} onClick={() => setSel({ kind: "obj", uid: o.uid })}>
                <span className="k">{o.origin === "machine" ? "M" : "H"}</span><span>{o.cls}</span>
                <span className="muted">{o.score != null ? o.score.toFixed(2) : ""} {o.correction !== "none" ? o.correction.replace("_", " ") : ""} {o.in_region === 0 ? "outside region" : ""}</span>
              </div>
            ))}
            {item.candidates.map((c) => (
              <div key={c.uid} className={`objrow ${sel?.uid === c.uid ? "sel" : ""}`} onClick={() => setSel({ kind: "cand", uid: c.uid })}>
                <span className="k">{c.status === "confirmed" ? "C" : "?"}</span><span>{c.kind} of lights</span>
                <span className="muted">{c.status === "unresolved" ? `${c.count_min}–${c.count_max}` : c.status.replace("_", " ")}</span>
              </div>
            ))}
          </div>
          <div className="card legend">
            <h3>Legend</h3>
            {([["machine", "M  detected by the model"], ["human", "H  drawn by an operator"], ["candidate", "?  night candidate, not counted"], ["confirmed", "C  candidate confirmed by an operator"]] as const).map(([k, l]) => (
              <div key={k}><svg width="36" height="10"><line x1="0" y1="5" x2="36" y2="5" stroke={STYLE[k].stroke} strokeWidth="2" strokeDasharray={STYLE[k].dash} /></svg> {l}</div>
            ))}
            {Object.keys(plates).length > 0 && <div><svg width="36" height="10"><line x1="0" y1="5" x2="36" y2="5" stroke="var(--plate)" strokeWidth="2" /></svg> plate found by the model (dashed: drawn by an operator)</div>}
            <div><svg width="36" height="10"><rect width="36" height="10" fill="url(#hatch)" stroke="var(--exclude)" /></svg> excluded area</div>
          </div>
        </aside>
      </div>
      {help && (
        <div className="dialog-bg" onClick={() => setHelp(false)}>
          <div className="dialog" onClick={(e) => e.stopPropagation()}>
            <h2>Keyboard shortcuts</h2>
            <table className="t"><tbody>
              {[["J / K", "Next / previous image"], ["A", "Approve this image"], ["B", "Draw a box for a missed vehicle"], ["Del", "Reject the selected object or candidate"],
                ["C", "Confirm the selected candidate as a vehicle"], ["G", "Group the chosen lights into one candidate"], ["1–5", "Set class: " + CLASSES.join(", ")],
                ["H", "Hide or show overlays"], ["Esc", "Cancel drawing or selection"], ["?", "This help"]].map(([k, v]) => <tr key={k}><td><kbd>{k}</kbd></td><td>{v}</td></tr>)}
            </tbody></table>
          </div>
        </div>
      )}
    </div>
  );
}
