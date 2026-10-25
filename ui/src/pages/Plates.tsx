import { useCallback, useEffect, useState } from "react";
import { api, ApiError, post } from "../api";

// Plate reads for plate recognition missions. Plate text is shown in bidi-isolated spans so Persian and Arabic
// characters keep their order next to Latin text. A country's reads are "evaluation only" until it passes an
// evaluation on reviewed plates and is enabled.

type Plate = {
  id: number; source_version_id: number; file: string; capture_utc: string | null; object_uid: string;
  vehicle_cls: string; vehicle_box: number[]; outcome: string; box: number[] | null; det_score: number | null;
  raw_text: string | null; chars: [string, number][]; machine_text: string | null; machine_display: string | null;
  machine_readability: string | null; layout: string | null; region_label: string | null; region_conf: number | null;
  reasons: string[]; country: string; evaluation_only: boolean; text: string | null; readability: string | null;
  review: string; revisions: number; operator_box: number[] | null;
  t?: number | null; event_key?: string | null;
};

const OUTCOME: Record<string, string> = {
  read: "Plate found", no_plate_found: "No plate found", vehicle_too_small: "Vehicle too small", plate_too_small: "Plate too small to read",
};
const REVIEW: Record<string, string> = {
  pending: "Not reviewed", confirmed: "Confirmed", corrected: "Corrected", unreadable: "Unreadable (operator)", no_plate: "No plate (operator)",
};
const LIMIT = 100;

export default function PlatesTab({ missionId }: { missionId: string }) {
  const [data, setData] = useState<any>(null);
  const [outcome, setOutcome] = useState("read");
  const [readability, setReadability] = useState("");
  const [review, setReview] = useState("");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);

  const load = useCallback(async () => {
    const q = new URLSearchParams({ offset: String(offset), limit: String(LIMIT) });
    if (outcome) q.set("outcome", outcome);
    if (readability) q.set("readability", readability);
    if (review) q.set("review", review);
    if (search) q.set("search", search);
    setData(await api(`/missions/${missionId}/plates?${q}`));
  }, [missionId, outcome, readability, review, search, offset]);
  useEffect(() => { load(); }, [load]);

  const act = async (fn: () => Promise<any>, ok?: (r: any) => string) => {
    setMsg(null);
    try { const r = await fn(); if (ok) setMsg({ kind: "ok", text: ok(r) }); await load(); }
    catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };

  if (!data) return <div className="muted">Loading…</div>;
  const c = data.country, ev = c.last_evaluation;
  const cnt = data.counts;
  return (
    <div className="grid" style={{ gap: 10 }}>
      <div className={`card ${c.enabled ? "" : "evalonly"}`}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>{c.name} plates: {c.enabled ? "enabled" : "evaluation only"}</h2>
          <div className="row">
            <button onClick={() => act(() => post(`/missions/${missionId}/plates/country/evaluate`),
              (r) => `Evaluation on ${r.reviewed_plates} reviewed plates: exact match ${(r.exact_match * 100).toFixed(1)}%, character error rate ${(r.cer * 100).toFixed(1)}%. ${r.passed ? "Passed." : "Not passed."}${r.finder_missed ? ` Plates the finder missed and you drew: ${r.finder_missed}.` : ""}`)}>Evaluate on reviewed plates</button>
            {!c.enabled && <button className="primary" disabled={!ev?.passed} onClick={() => act(() => post(`/missions/${missionId}/plates/country/enable`, { enabled: true }))}>Enable {c.name}</button>}
            {c.enabled && <button className="danger" onClick={() => act(() => post(`/missions/${missionId}/plates/country/enable`, { enabled: false }))}>Back to evaluation only</button>}
          </div>
        </div>
        <p style={{ margin: "6px 0" }}>
          {c.reader_covers ? `The installed reader covers ${c.script}.` : `The installed reader cannot read ${c.script}: plates are found but not read.`}{" "}
          {!c.enabled && "Reads are shown and exported marked “evaluation only” until an evaluation on plates you reviewed passes and you enable the country."}
        </p>
        <p className="muted" style={{ margin: 0 }}>
          Layouts used: {c.layouts.map((l: any) => `${l.name} (${l.note})`).join("; ") || "none"}.{" "}
          {ev ? <>Last evaluation {ev.at} by {ev.by}: {ev.reviewed_plates} plates, exact match {(ev.exact_match * 100).toFixed(1)}%, CER {(ev.cer * 100).toFixed(1)}%, machine reads of plates you marked unreadable: {ev.machine_guesses_on_unreadable}. Needs at least {ev.thresholds.min_plates} plates, {Math.round(ev.thresholds.min_exact * 100)}% exact, CER at most {Math.round(ev.thresholds.max_cer * 100)}%, and no reads of unreadable plates.</> : "No evaluation yet."}
        </p>
      </div>
      <div className="grid cols-4">
        {(["read", "no_plate_found", "plate_too_small", "vehicle_too_small"] as const).map((k) =>
          <div key={k} className="card stat"><div className="n">{cnt.outcome[k] ?? 0}</div><div className="l">{OUTCOME[k]}</div></div>)}
      </div>
      <div className="card row">
        <select value={outcome} onChange={(e) => { setOffset(0); setOutcome(e.target.value); }} aria-label="Outcome">
          <option value="">All vehicles</option>{Object.entries(OUTCOME).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <select value={readability} onChange={(e) => { setOffset(0); setReadability(e.target.value); }} aria-label="Readability">
          <option value="">Any readability</option><option value="full">Full ({cnt.readability.full ?? 0})</option>
          <option value="partial">Partial ({cnt.readability.partial ?? 0})</option><option value="unreadable">Unreadable ({cnt.readability.unreadable ?? 0})</option>
        </select>
        <select value={review} onChange={(e) => { setOffset(0); setReview(e.target.value); }} aria-label="Review">
          <option value="">Any review</option>{Object.entries(REVIEW).map(([k, v]) => <option key={k} value={k}>{v} ({cnt.review[k] ?? 0})</option>)}
        </select>
        <input placeholder="Plate text contains" value={search} onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
        <span className="muted">{data.total} vehicles</span>
        <span style={{ flex: 1 }} />
        <button onClick={() => act(() => post(`/missions/${missionId}/plates/export`, {}), (r) => `Exported: ${r.files.join(", ")}`)}>Export plates</button>
      </div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
      {data.total === 0 && <div className="card muted">No vehicles match. Plates are read after vehicle detection, when you scan and process.</div>}
      <div className="grid" style={{ gap: 8 }}>
        {data.plates.map((p: Plate) => <PlateRow key={p.id} p={p} missionId={missionId} act={act} />)}
      </div>
      {data.total > LIMIT && (
        <div className="row">
          <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>Previous</button>
          <span className="muted">{offset + 1}–{Math.min(offset + LIMIT, data.total)} of {data.total}</span>
          <button disabled={offset + LIMIT >= data.total} onClick={() => setOffset(offset + LIMIT)}>Next</button>
        </div>
      )}
    </div>
  );
}

function PlateRow({ p, missionId, act }: { p: Plate; missionId: string; act: (fn: () => Promise<any>) => void }) {
  const [text, setText] = useState("");
  const rv = (action: string, extra: Record<string, unknown> = {}) =>
    act(() => post(`/missions/${missionId}/plates/${p.id}/revise`, { action, ...extra }));
  const vbox = p.vehicle_box.map((v) => v.toFixed(1)).join(",");
  const eventId = p.event_key ? Number(p.object_uid.slice(1)) : null;   // video: the vehicle event
  const openHref = eventId ? `#/m/${missionId}/events/${eventId}` : `#/m/${missionId}/i/${p.source_version_id}`;
  return (
    <div className="card plate">
      <a href={openHref} title={eventId ? "Open the event" : "Open the image"}>
        <img className="veh" src={eventId ? `/api/missions/${missionId}/events/${eventId}/thumb?h=90`
          : `/api/missions/${missionId}/crops/${p.source_version_id}?box=${vbox}&h=90`} alt={`${p.vehicle_cls} in ${p.file}`} loading="lazy" />
      </a>
      <div className="grid" style={{ gap: 4, alignContent: "start" }}>
        {p.operator_box ? <><img className="pl" src={`/api/missions/${missionId}/crops/${p.source_version_id}?box=${p.operator_box.join(",")}&h=56`} alt="plate (box drawn by an operator)" loading="lazy" /><span className="muted" style={{ fontSize: 11.5 }}>box drawn by an operator</span></>
          : p.box ? <img className="pl" src={`/api/missions/${missionId}/plates/${p.id}/crop?h=56`} alt="plate" loading="lazy" /> : <span className="muted">{OUTCOME[p.outcome]}</span>}
        {p.chars.length > 0 && p.raw_text != null && (
          <div className="chars" title="Each character the reader produced, shaded by its confidence">
            {p.chars.filter(([ch, cf]) => !(ch === "_" && cf >= 0.6)).map(([ch, cf], i) =>
              <span key={i} style={{ opacity: 0.35 + 0.65 * cf, borderColor: cf < 0.6 ? "var(--bad)" : "var(--line)" }}>{ch === "_" ? "␣" : ch}<small>{Math.round(cf * 100)}</small></span>)}
          </div>
        )}
      </div>
      <div className="grid" style={{ gap: 4, alignContent: "start" }}>
        <div className="row">
          <span className="platetext"><bdi>{p.text ?? (p.readability === "unreadable" ? "unreadable" : "–")}</bdi></span>
          {p.readability && <span className={`badge ${p.readability === "full" ? "approved" : p.readability === "partial" ? "bulk_approved_unreviewed" : "failed"}`}>{p.readability}</span>}
          <span className="badge">{REVIEW[p.review]}</span>
          {p.evaluation_only && <span className="badge flag">evaluation only</span>}
        </div>
        <div className="muted" style={{ fontSize: 12.5 }}>
          Machine: <bdi className="mono">{p.machine_display ?? "–"}</bdi>{p.raw_text != null && <> · raw <bdi className="mono">{p.raw_text || "(empty)"}</bdi></>}
          {p.layout && <> · layout {p.layout}</>}{p.det_score != null && <> · finder {p.det_score.toFixed(2)}</>}
          {p.region_label && <> · reader's country guess {p.region_label} ({Math.round((p.region_conf ?? 0) * 100)}%)</>}
        </div>
        {p.reasons.length > 0 && <ul className="reasons" style={{ margin: 0 }}>{p.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>}
        <div className="mono muted" style={{ fontSize: 11.5 }}>{p.file}{p.event_key && <> · {p.event_key} at {p.t?.toFixed(1)} s</>} · {p.vehicle_cls}</div>
      </div>
      <div className="grid" style={{ gap: 4, alignContent: "start" }}>
        <div className="row" style={{ gap: 4 }}>
          <button disabled={p.machine_readability !== "full"} onClick={() => rv("confirm")} title="The machine read is right">Confirm</button>
          <button onClick={() => rv("unreadable")}>Unreadable</button>
          <button onClick={() => rv("no_plate")} title="Not a plate, or no plate visible">No plate</button>
          {p.review !== "pending" && <button onClick={() => rv("reopen")}>Reopen</button>}
        </div>
        <div className="row" style={{ gap: 4 }}>
          <input dir="auto" placeholder="Type the plate as read" value={text} onChange={(e) => setText(e.target.value)} style={{ width: 170 }} />
          <button className="primary" disabled={!text.trim()} onClick={() => { rv("correct", { text }); setText(""); }}>Save</button>
        </div>
        <span className="muted" style={{ fontSize: 12 }}>Type only what you can read. If a character is unclear, mark the plate unreadable.
          {eventId ? <> To check other moments, <a href={openHref}>open the event</a> and play it.</>
            : <> To draw a missed plate's box, <a href={openHref}>open the image</a> and select the vehicle.</>}</span>
      </div>
    </div>
  );
}
