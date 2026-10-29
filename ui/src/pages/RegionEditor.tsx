import { useEffect, useRef, useState } from "react";
import { api, ApiError, AreaProposal, AreaTask, post, Region, Sam3Status } from "../api";

type Kind = "include" | "exclude";
type Data = { width: number; height: number; representatives: Record<string, number[]>; regions: Region[] };

const pointsAttr = (poly: [number, number][]) => poly.map((p) => p.join(",")).join(" ");
const pct = (x: number) => (x >= 0.1 ? `${Math.round(x * 100)}%` : `${(x * 100).toFixed(1)}%`);

// Points are stored in the view's original pixels; display size and zoom never change them (VIEW-03).
export default function RegionEditor({ missionId, viewId }: { missionId: string; viewId: number }) {
  const [data, setData] = useState<Data | null>(null);
  const [bgId, setBgId] = useState<number | null>(null);
  const [mode, setMode] = useState<"find" | "draw">("find");
  const [kind, setKind] = useState<Kind>("include");
  const [name, setName] = useState("");
  const [pts, setPts] = useState<[number, number][]>([]);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sam, setSam] = useState<Sam3Status | null>(null);
  const [task, setTask] = useState<AreaTask | null>(null);
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [focus, setFocus] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const load = async () => {
    const d = await api(`/missions/${missionId}/views/${viewId}/regions`);
    setData(d);
    if (bgId == null) setBgId((d.representatives.colour ?? Object.values(d.representatives)[0] ?? [null])[0]);
  };
  useEffect(() => { load(); }, [missionId, viewId]);
  useEffect(() => {
    api<Sam3Status>("/sam3").then((s) => { setSam(s); if (!s.installed) setMode("draw"); }).catch(() => setMode("draw"));
  }, []);

  // Follow a running search until it ends.
  const running = task?.state === "running";
  useEffect(() => {
    if (!task || !running) return;
    const t = setInterval(async () => {
      try {
        const next = await api<AreaTask>(`/missions/${missionId}/find-areas/${task.id}`);
        setTask(next);
        if (next.state === "finished") setPicked(new Set(next.proposals.map((_, i) => i)));
      } catch (e) { setTask(null); setErr((e as ApiError).message); }
    }, 700);
    return () => clearInterval(t);
  }, [task?.id, running]);

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

  const proposals = task?.state === "finished" ? task.proposals : [];
  const find = async (prompts: string[], k: Kind) => {
    setErr(null);
    setPicked(new Set());
    try {
      setTask(await post<AreaTask>(`/missions/${missionId}/views/${viewId}/find-areas`, { prompts, kind: k, image_id: bgId }));
    } catch (e) { setErr((e as ApiError).message); }
  };
  const accept = async () => {
    setErr(null);
    const chosen = proposals.filter((_, i) => picked.has(i));
    try {
      await post(`/missions/${missionId}/views/${viewId}/regions/batch`, {
        regions: chosen.map((p) => ({ kind: p.kind, polygon: p.polygon, name: `${p.prompt} (found by SAM 3.1)` })),
      });
      setTask(null);
      setPicked(new Set());
      load();
    } catch (e) { setErr((e as ApiError).message); }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (mode !== "draw" || ["INPUT", "TEXTAREA"].includes((e.target as HTMLElement).tagName)) return;
      if (e.key === "Escape") setPts([]);
      if (e.key === "Backspace") setPts((p) => p.slice(0, -1));
      if (e.key === "Enter" && pts.length >= 3) save();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!data) return <div className="page muted">Loading…</div>;
  const sw = Math.max(1.5, data.width / 640);
  const drawing = mode === "draw";
  return (
    <div className="page" style={{ maxWidth: "none" }}>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <a href={`#/m/${missionId}/views`}>← Camera views</a>
        <div className="row">
          <span className="muted">Background image:</span>
          {Object.entries(data.representatives).map(([c, ids]) => ids.map((id) => (
            <button key={id} className={bgId === id ? "primary" : ""} disabled={running} onClick={() => { setBgId(id); setTask(null); }}>{c}</button>
          )))}
        </div>
      </div>
      <div className="inv">
        <div className={`stage${running ? " scanning" : ""}`}>
          {bgId && <img src={`/api/missions/${missionId}/images/${bgId}`} alt="" draggable={false} />}
          <svg ref={svgRef} viewBox={`0 0 ${data.width} ${data.height}`} preserveAspectRatio="xMidYMid meet" style={{ cursor: drawing ? "crosshair" : "default" }}
            onClick={(e) => drawing && setPts([...pts, toImage(e)])} onMouseMove={(e) => drawing && setHover(toImage(e))} onMouseLeave={() => setHover(null)}>
            <defs>
              <pattern id="hatch2" width={10 * sw} height={10 * sw} patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
                <line x1="0" y1="0" x2="0" y2={10 * sw} stroke="var(--exclude)" strokeWidth={sw} />
              </pattern>
              <pattern id="hatch-found" width={8 * sw} height={8 * sw} patternTransform="rotate(-45)" patternUnits="userSpaceOnUse">
                <line x1="0" y1="0" x2="0" y2={8 * sw} stroke="var(--exclude)" strokeWidth={sw * 0.8} strokeOpacity=".8" />
              </pattern>
            </defs>
            {data.regions.map((r) => (
              <g key={r.uid} opacity={proposals.length ? 0.45 : 1}>
                <polygon points={pointsAttr(r.polygon)} fill={r.kind === "exclude" ? "url(#hatch2)" : "rgba(114,183,255,0.16)"}
                  stroke={r.kind === "exclude" ? "var(--exclude)" : "var(--include)"} strokeWidth={sw * 1.5} strokeDasharray={r.kind === "exclude" ? `${6 * sw} ${3 * sw}` : undefined} />
                <text x={r.polygon[0][0]} y={r.polygon[0][1] - 4 * sw} fill="#fff" fontSize={12 * sw} paintOrder="stroke" stroke="#000" strokeWidth={3 * sw}>
                  {r.kind === "exclude" ? "Excluded" : "Included"}{r.name ? `: ${r.name}` : ""}</text>
              </g>
            ))}
            {proposals.map((p, i) => {
              const on = picked.has(i);
              return (
                <polygon key={`p${i}`} className={`found${on ? " on" : ""}${focus === i ? " focus" : ""}`} points={pointsAttr(p.polygon)}
                  fill={p.kind === "exclude" ? "url(#hatch-found)" : undefined}
                  strokeWidth={sw * (focus === i ? 3 : 2)} strokeDasharray={on ? undefined : `${5 * sw} ${4 * sw}`}
                  data-kind={p.kind} onMouseEnter={() => setFocus(i)} onMouseLeave={() => setFocus(null)}
                  onClick={() => setPicked((s) => { const n = new Set(s); n.has(i) ? n.delete(i) : n.add(i); return n; })} />
              );
            })}
            {drawing && pts.length > 0 && (
              <polyline points={pointsAttr([...pts, ...(hover ? [hover] : [])])} fill="rgba(255,255,255,.12)"
                stroke={kind === "exclude" ? "var(--exclude)" : "var(--include)"} strokeWidth={sw * 2} />
            )}
            {drawing && pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={4 * sw} fill="#fff" stroke="#000" strokeWidth={sw} />)}
          </svg>
          {running && <div className="scanbar" aria-hidden="true" />}
        </div>
        <aside className="side">
          <div className="card">
            <div className="seg" role="tablist" aria-label="How to make a region">
              <button role="tab" aria-selected={mode === "find"} className={mode === "find" ? "on" : ""} onClick={() => setMode("find")}>Find by name</button>
              <button role="tab" aria-selected={drawing} className={drawing ? "on" : ""} onClick={() => setMode("draw")}>Draw by hand</button>
            </div>
            {mode === "find" ? (
              <FindPanel sam={sam} kind={kind} setKind={setKind} task={task} busy={running} onFind={find}
                proposals={proposals} picked={picked} setPicked={setPicked} focus={focus} setFocus={setFocus}
                onAccept={accept} onDiscard={() => { setTask(null); setPicked(new Set()); }} />
            ) : (
              <>
                <p className="muted">Click on the image to add corners. <kbd>Backspace</kbd> removes the last one, <kbd>Enter</kbd> saves, <kbd>Esc</kbd> clears.</p>
                <KindSwitch kind={kind} setKind={setKind} />
                <label className="field" style={{ marginTop: 12 }}><span>Name (optional)</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. parking lot" /></label>
                <div className="mono muted">{pts.length} corner(s){hover ? ` · pointer at ${hover[0]}, ${hover[1]} px` : ""}</div>
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="primary" disabled={pts.length < 3} onClick={save}>Save region</button>
                  <button onClick={() => setPts([])}>Clear</button>
                </div>
              </>
            )}
            {err && <div className="msg err" role="alert">{err}</div>}
          </div>
          <div className="card">
            <h2>Regions of this view</h2>
            <p className="muted">Exclusion wins over inclusion. With no inclusion region the whole frame is analysed. A vehicle counts when the middle of its bottom edge is inside, and parts of the frame outside the regions are not searched closely.</p>
            {data.regions.length === 0 && <p className="muted">None yet.</p>}
            {data.regions.map((r) => (
              <div key={r.uid} className="row" style={{ justifyContent: "space-between", padding: "4px 0" }}>
                <span><span className={`badge kind-${r.kind}`}>{r.kind}</span> {r.name ?? <span className="muted">unnamed</span>} <span className="muted">({r.polygon.length} corners)</span></span>
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

function KindSwitch({ kind, setKind, disabled }: { kind: Kind; setKind: (k: Kind) => void; disabled?: boolean }) {
  return (
    <div className="row" role="radiogroup" aria-label="Region type">
      <label className="row"><input type="radio" disabled={disabled} checked={kind === "include"} onChange={() => setKind("include")} /> Include (road to analyse)</label>
      <label className="row"><input type="radio" disabled={disabled} checked={kind === "exclude"} onChange={() => setKind("exclude")} /> Exclude</label>
    </div>
  );
}

function FindPanel(p: {
  sam: Sam3Status | null; kind: Kind; setKind: (k: Kind) => void; task: AreaTask | null; busy: boolean;
  onFind: (prompts: string[], kind: Kind) => void; proposals: AreaProposal[]; picked: Set<number>;
  setPicked: (s: Set<number>) => void; focus: number | null; setFocus: (i: number | null) => void;
  onAccept: () => void; onDiscard: () => void;
}) {
  const { sam, kind } = p;
  const [chosen, setChosen] = useState<Record<Kind, string[]>>({ include: ["road"], exclude: [] });
  const [extra, setExtra] = useState("");
  if (!sam) return <p className="muted">Checking for SAM 3.1…</p>;
  if (!sam.installed) {
    return (
      <div className="msg warn">
        <b>SAM 3.1 is not installed on this computer.</b>
        <p style={{ margin: "4px 0 0" }}>{sam.reason} Draw the regions by hand, or install the full LanePulse package, which includes it.</p>
      </div>
    );
  }
  const words = chosen[kind];
  const suggestions = [...new Set([...(sam.suggestions?.[kind] ?? []), ...words])];
  const toggle = (w: string) => setChosen((c) => ({ ...c, [kind]: c[kind].includes(w) ? c[kind].filter((x) => x !== w) : [...c[kind], w] }));
  const addExtra = () => {
    const w = extra.trim().toLowerCase();
    if (w && !words.includes(w)) setChosen((c) => ({ ...c, [kind]: [...c[kind], w] }));
    setExtra("");
  };
  const t = p.task;
  const n = p.picked.size;
  return (
    <>
      <p className="muted">Name what to analyse, or what to leave out. SAM 3.1 outlines it on this image, and you choose which outlines become regions.</p>
      <KindSwitch kind={kind} setKind={p.setKind} disabled={p.busy} />
      <div className="chips" role="group" aria-label="What to look for" style={{ marginTop: 12 }}>
        {suggestions.map((w) => (
          <button key={w} className={`chip${words.includes(w) ? " on" : ""}`} aria-pressed={words.includes(w)} disabled={p.busy} onClick={() => toggle(w)}>{w}</button>
        ))}
      </div>
      <form className="row" style={{ marginTop: 10 }} onSubmit={(e) => { e.preventDefault(); addExtra(); }}>
        <input value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="Other words, e.g. bus lane" maxLength={60} disabled={p.busy} style={{ flex: 1 }} aria-label="Other words to look for" />
        <button type="submit" disabled={!extra.trim() || p.busy}>Add</button>
      </form>
      <div className="row" style={{ marginTop: 12 }}>
        <button className="primary" disabled={!words.length || p.busy} onClick={() => p.onFind(words, kind)}>
          {p.busy ? "Looking…" : `Find ${kind === "include" ? "areas to analyse" : "areas to leave out"}`}
        </button>
      </div>
      {t?.state === "running" && (
        <div className="found-status" role="status" aria-live="polite">
          <span className="pulse" aria-hidden="true" /> {cap(t.stage)} <span className="muted">· {Math.round(t.elapsed_s)} s</span>
          {t.stage.startsWith("loading") && <div className="muted" style={{ marginTop: 4 }}>The first search loads the model, which takes up to a minute. Later searches are quicker.</div>}
        </div>
      )}
      {t?.state === "failed" && <div className="msg err" role="alert">SAM 3.1 stopped: {t.error}</div>}
      {t?.state === "finished" && (
        <div className="found-list">
          {p.proposals.length === 0 ? (
            <p className="muted" style={{ marginTop: 12 }}>Nothing matching {t.prompts.map((x) => `“${x}”`).join(", ")} was found in {Math.round(t.elapsed_s)} s. Try other words, another background image, or draw the region by hand.</p>
          ) : (
            <>
              <h3 style={{ marginTop: 14 }}>Found in {Math.round(t.elapsed_s)} s</h3>
              <ul>
                {groups(p.proposals).map((g) => {
                  const on = g.idx.filter((i) => p.picked.has(i)).length;
                  const share = g.idx.reduce((a, i) => a + p.proposals[i].area_share, 0);
                  const hot = p.focus !== null && g.idx.includes(p.focus);
                  return (
                    <li key={g.key} className={hot ? "focus" : ""} onMouseEnter={() => p.setFocus(g.idx[0])} onMouseLeave={() => p.setFocus(null)}>
                      <label className="row">
                        <input type="checkbox" checked={on === g.idx.length} ref={(el) => { if (el) el.indeterminate = on > 0 && on < g.idx.length; }}
                          onChange={() => { const s = new Set(p.picked); g.idx.forEach((i) => (on === g.idx.length ? s.delete(i) : s.add(i))); p.setPicked(s); }} />
                        <span className={`badge kind-${g.kind}`}>{g.kind}</span>
                        <span className="grow">{g.prompt}{g.idx.length > 1 && <small className="muted">{on < g.idx.length ? `${on} of ` : ""}{g.idx.length} outlines</small>}</span>
                        <span className="muted num">{pct(share)} of image</span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              <p className="muted">Click an outline on the image to leave it out or take it back. Saved outlines are ordinary regions: you can delete them, or draw more by hand.</p>
              <div className="row">
                <button className="primary" disabled={!n} onClick={p.onAccept}>{n ? `Add ${n} region${n > 1 ? "s" : ""}` : "Choose an outline"}</button>
                <button onClick={p.onDiscard}>Discard</button>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// One row per word and kind; single outlines are switched on and off on the image itself.
function groups(ps: AreaProposal[]) {
  const out: { key: string; kind: string; prompt: string; idx: number[] }[] = [];
  ps.forEach((a, i) => {
    const key = `${a.kind}|${a.prompt}`;
    const g = out.find((x) => x.key === key);
    if (g) g.idx.push(i); else out.push({ key, kind: a.kind, prompt: a.prompt, idx: [i] });
  });
  return out;
}
