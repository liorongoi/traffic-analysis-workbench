import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, Box, post } from "../api";

// Night review: candidates from every night or dusk image as a grid of crops, most uncertain first.
// One key per answer. Answers are the normal candidate decisions, so they also teach the background
// ("decisions at a place") and give the accuracy numbers at the top.

type QItem = {
  version_id: number; file: string; capture_local: string | null; condition: string; view_id: number | null;
  uid: string; kind: string; box: Box; score: number | null; status: string; band: string | null;
  recurrence: number | null; flags: string[]; alternatives: string[]; count_min: number; count_max: number; cls: string | null;
};
type Rate = { vehicle: number; not_vehicle: number; open: number; answered: number; vehicle_share: number | null };
type Accuracy = { overall: Rate; by_band: Record<string, Rate>; by_kind: Record<string, Rate>;
  by_score: Record<string, Rate>; by_flag: Record<string, Rate>; note: string };

const pct = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const STATUS: Record<string, string> = { unresolved: "", confirmed: "vehicle", rejected: "not a vehicle", persistent_light: "fixed light" };
const PER = 60;

function RateTable({ title, rows }: { title: string; rows: Record<string, Rate> }) {
  const keys = Object.keys(rows);
  if (!keys.length) return null;
  return (
    <table className="t">
      <thead><tr><th>{title}</th><th>Vehicles</th><th>Not vehicles</th><th>Share vehicles</th><th>Not answered</th></tr></thead>
      <tbody>{keys.map((k) => <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{rows[k].vehicle}</td><td>{rows[k].not_vehicle}</td>
        <td>{pct(rows[k].vehicle_share)}</td><td className="muted">{rows[k].open}</td></tr>)}</tbody>
    </table>
  );
}

export default function NightReviewTab({ missionId }: { missionId: string }) {
  const [views, setViews] = useState<any[]>([]);
  const [view, setView] = useState("");
  const [band, setBand] = useState("");
  const [flag, setFlag] = useState("");
  const [status, setStatus] = useState("open");
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ total: number; items: QItem[] } | null>(null);
  const [acc, setAcc] = useState<Accuracy | null>(null);
  const [sel, setSel] = useState(0);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [showAcc, setShowAcc] = useState(false);
  const gridRef = useRef<HTMLDivElement>(null);

  // Only filters that are set go in the query: an empty view is "all views", not a view id.
  const qs = new URLSearchParams(Object.entries({ view, band, flag }).filter(([, v]) => v !== "")).toString();
  const load = useCallback(async () => {
    try {
      const r = await api(`/missions/${missionId}/night/queue?${qs}&status=${status}&offset=${page * PER}&limit=${PER}`);
      setData(r);
      setSel(0);
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  }, [missionId, qs, status, page]);
  const loadAcc = useCallback(() => api(`/missions/${missionId}/night/accuracy${view ? `?view=${view}` : ""}`).then(setAcc), [missionId, view]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { loadAcc(); }, [loadAcc]);
  useEffect(() => { api(`/missions/${missionId}/views`).then((r) => setViews(r.views)); }, [missionId]);

  const items = data?.items ?? [];
  const cur = items[sel];

  const answer = async (it: QItem, action: string) => {
    setMsg(null);
    try {
      await post(`/missions/${missionId}/items/${it.version_id}/candidates/${it.uid}`,
        action === "confirm" ? { action, cls: it.cls ?? "car" } : { action });
      const st = { confirm: "confirmed", reject: "rejected", persistent: "persistent_light", restore: "unresolved" }[action]!;
      setData((d) => d && { ...d, items: d.items.map((x) => (x.uid === it.uid && x.version_id === it.version_id ? { ...x, status: st } : x)) });
      if (action !== "restore") setSel((s) => Math.min(s + 1, items.length - 1));
      loadAcc();
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest("input,select,textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      const perRow = Math.max(1, Math.floor((gridRef.current?.clientWidth ?? 800) / 176));
      if (k === "arrowright") setSel((s) => Math.min(s + 1, items.length - 1));
      else if (k === "arrowleft") setSel((s) => Math.max(s - 1, 0));
      else if (k === "arrowdown") setSel((s) => Math.min(s + perRow, items.length - 1));
      else if (k === "arrowup") setSel((s) => Math.max(s - perRow, 0));
      else if (!cur) return;
      else if (k === "v") answer(cur, "confirm");
      else if (k === "n" || k === "delete" || k === "backspace") answer(cur, "reject");
      else if (k === "f") answer(cur, "persistent");
      else if (k === "u") answer(cur, "restore");
      else if (k === "enter") window.location.hash = `#/m/${missionId}/i/${cur.version_id}`;
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  useEffect(() => {
    gridRef.current?.querySelector<HTMLElement>(`[data-i="${sel}"]`)?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  const o = acc?.overall;
  const flags = acc ? Object.keys(acc.by_flag).filter((f) => f !== "no_flags") : [];
  const pages = data ? Math.max(1, Math.ceil(data.total / PER)) : 1;
  const reset = (fn: () => void) => { setPage(0); fn(); };

  return (
    <div className="grid" style={{ gap: 10 }}>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Night candidates</h2>
          <button onClick={() => setShowAcc(!showAcc)}>{showAcc ? "Hide" : "Show"} accuracy details</button>
        </div>
        {o && (
          <div className="grid cols-4" style={{ marginTop: 10 }}>
            <div className="card stat"><div className="n">{pct(o.vehicle_share)}</div><div className="l">of answered candidates were vehicles</div></div>
            <div className="card stat"><div className="n">{o.vehicle}</div><div className="l">Vehicles</div></div>
            <div className="card stat"><div className="n">{o.not_vehicle}</div><div className="l">Not vehicles (incl. fixed lights)</div></div>
            <div className="card stat"><div className="n">{o.open}</div><div className="l">Not answered yet</div></div>
          </div>
        )}
        {showAcc && acc && (
          <div className="grid" style={{ gap: 8, marginTop: 10 }}>
            <div className="tablewrap"><RateTable title="Distance" rows={acc.by_band} /></div>
            <div className="tablewrap"><RateTable title="Score" rows={acc.by_score} /></div>
            <div className="tablewrap"><RateTable title="Kind" rows={acc.by_kind} /></div>
            <div className="tablewrap"><RateTable title="Flag" rows={acc.by_flag} /></div>
          </div>
        )}
        {acc && <p className="muted" style={{ marginBottom: 0 }}>{acc.note}</p>}
      </div>

      <div className="card">
        <div className="row">
          <select value={view} onChange={(e) => reset(() => setView(e.target.value))}>
            <option value="">All camera views</option>
            {views.map((v) => <option key={v.id} value={v.id}>{v.name ?? `View ${v.id}`}</option>)}
          </select>
          <select value={band} onChange={(e) => reset(() => setBand(e.target.value))}>
            <option value="">Any distance</option><option value="far">Far</option><option value="middle">Middle</option>
            <option value="close">Close</option><option value="none">No distance</option>
          </select>
          <select value={flag} onChange={(e) => reset(() => setFlag(e.target.value))}>
            <option value="">Any flag</option>
            {flags.map((f) => <option key={f} value={f}>{f.replace(/_/g, " ")}</option>)}
          </select>
          <select value={status} onChange={(e) => reset(() => setStatus(e.target.value))}>
            <option value="open">Not answered (most uncertain first)</option><option value="answered">Answered</option><option value="all">All</option>
          </select>
          <span className="muted">{data ? `${data.total} candidates` : ""}</span>
          {pages > 1 && <>
            <button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
            <span className="muted">Page {page + 1} of {pages}</span>
            <button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button>
          </>}
        </div>
        <p className="muted" style={{ margin: "8px 0" }}>
          <kbd>V</kbd> vehicle · <kbd>N</kbd> or <kbd>Del</kbd> not a vehicle · <kbd>F</kbd> fixed light (lamp, sign) ·
          <kbd>U</kbd> undo · arrows move · <kbd>Enter</kbd> open the whole image. The orange box marks the candidate.
        </p>
        {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
        {data && !items.length && <div className="muted">Nothing to review with these filters.</div>}
        <div className="nightgrid" ref={gridRef}>
          {items.map((it, i) => {
            const w = it.box[2] - it.box[0], h = it.box[3] - it.box[1];
            const ctx = Math.round(Math.min(60, Math.max(24, Math.max(w, h))));
            return (
              <div key={`${it.version_id}-${it.uid}`} data-i={i} className={`ntile${i === sel ? " on" : ""} s-${it.status}`}
                   onClick={() => setSel(i)} onDoubleClick={() => { window.location.hash = `#/m/${missionId}/i/${it.version_id}`; }}>
                <img loading="lazy" alt="" src={`/api/missions/${missionId}/crops/${it.version_id}?box=${it.box.map((v) => v.toFixed(1)).join(",")}&h=120&context=${ctx}&mark=true`} />
                <div className="nmeta">
                  <span>{it.score == null ? "–" : it.score.toFixed(2)}</span>
                  <span>{it.band ?? ""}</span>
                  <span>{it.kind.replace(/_/g, " ")}</span>
                </div>
                {it.flags.length > 0 && <div className="nflags" title={it.flags.join(", ")}>{it.flags.map((f) => f.replace(/_/g, " ")).join(" · ")}</div>}
                {STATUS[it.status] && <div className="nstatus">{STATUS[it.status]}</div>}
                {i === sel && <div className="row nact">
                  <button className="primary" onClick={(e) => { e.stopPropagation(); answer(it, "confirm"); }}>V</button>
                  <button onClick={(e) => { e.stopPropagation(); answer(it, "reject"); }}>N</button>
                  <button onClick={(e) => { e.stopPropagation(); answer(it, "persistent"); }}>F</button>
                </div>}
              </div>
            );
          })}
        </div>
        {cur && <div className="muted mono bidi" style={{ marginTop: 8 }}>{cur.file} · {cur.capture_local ?? ""} · {cur.condition}</div>}
      </div>
    </div>
  );
}
