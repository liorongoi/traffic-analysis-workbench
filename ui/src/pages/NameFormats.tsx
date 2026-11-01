import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, post } from "../api";

// Filename formats: the operator shows the app how their camera names files, so it can read the capture time.
type Format = { name: string; template: string; zone: string };
type Token = { token: string; label: string };
type TryRow = { name: string; ok: boolean; reason?: string; utc?: string; local?: string; precision?: string;
  parts?: Record<string, string> };
type TryResult = { ok: boolean; error?: string; precision?: string; results: TryRow[];
  folder?: { error?: string; checked?: number; matched?: number; not_matched_examples?: string[] } };

const PART_LABEL: Record<string, string> = { year: "Year", year2: "Year", month: "Month", day: "Day", hour: "Hour",
  minute: "Minute", second: "Second", camera: "Camera", suffix: "ID / other" };
const PART_ORDER = ["year", "year2", "month", "day", "hour", "minute", "second", "camera", "suffix"];
const blank: Format = { name: "", template: "", zone: "UTC" };

export default function NameFormats() {
  const [formats, setFormats] = useState<Format[] | null>(null);
  const [builtin, setBuiltin] = useState<{ name: string; description: string }[]>([]);
  const [tokens, setTokens] = useState<Token[]>([]);
  const [zones, setZones] = useState<string[]>([]);
  const [editing, setEditing] = useState<number | null>(null);   // index in formats, or -1 for a new one
  const [draft, setDraft] = useState<Format>(blank);
  const [examples, setExamples] = useState("");
  const [folder, setFolder] = useState("");
  const [subdirs, setSubdirs] = useState(true);
  const [tried, setTried] = useState<TryResult | null>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [zoneFilter, setZoneFilter] = useState("");
  const tplRef = useRef<HTMLInputElement>(null);

  const load = () => api("/name-formats").then((r) => { setFormats(r.formats); setBuiltin(r.builtin); setTokens(r.tokens); })
    .catch((e) => setMsg({ kind: "err", text: (e as Error).message }));
  useEffect(() => { load(); api("/reference").then((r) => setZones(r.zones ?? [])).catch(() => {}); }, []);

  const exampleList = useMemo(() => examples.split("\n").map((s) => s.trim()).filter(Boolean), [examples]);

  // Live preview as the template, zone or examples change.
  useEffect(() => {
    if (editing === null || !draft.template.trim()) { setTried(null); return; }
    const t = setTimeout(() => {
      post<TryResult>("/name-formats/try", { template: draft.template, zone: draft.zone, examples: exampleList })
        .then(setTried).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [draft.template, draft.zone, exampleList, editing]);

  const start = (i: number) => {
    setMsg(null);
    setEditing(i);
    setDraft(i >= 0 && formats ? { ...formats[i] } : blank);
    setExamples("");
    setTried(null);
  };

  const guess = async () => {
    if (!exampleList.length) return;
    const r = await post<{ template: string }>("/name-formats/suggest", { examples: exampleList });
    if (r.template) setDraft((d) => ({ ...d, template: r.template }));
    else setMsg({ kind: "err", text: "No date and time found in that name. Build the format with the parts below." });
  };

  const insert = (tok: string) => {
    const el = tplRef.current;
    const pos = el?.selectionStart ?? draft.template.length;
    const end = el?.selectionEnd ?? pos;
    const next = draft.template.slice(0, pos) + tok + draft.template.slice(end);
    setDraft((d) => ({ ...d, template: next }));
    requestAnimationFrame(() => { el?.focus(); el?.setSelectionRange(pos + tok.length, pos + tok.length); });
  };

  const checkFolder = async () => {
    setBusy(true);
    try {
      setTried(await post<TryResult>("/name-formats/try", { template: draft.template, zone: draft.zone, examples: exampleList,
        folder, include_subdirs: subdirs }));
    } finally { setBusy(false); }
  };

  const saveAll = async (list: Format[], done: string) => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api<{ formats: Format[] }>("/name-formats", { method: "PUT", body: { formats: list } });
      setFormats(r.formats);
      setEditing(null);
      setMsg({ kind: "ok", text: done });
    } catch (e) {
      setMsg({ kind: "err", text: (e as ApiError).message });
    } finally { setBusy(false); }
  };

  const save = () => {
    if (!formats) return;
    const list = [...formats];
    if (editing === -1) list.push(draft); else if (editing !== null) list[editing] = draft;
    saveAll(list, `Saved “${draft.name}”. Press Scan and process on your mission: images skipped before are read again.`);
  };

  const remove = (i: number) => {
    if (!formats || !confirm(`Remove the format “${formats[i].name}”? Images already read keep their time.`)) return;
    saveAll(formats.filter((_, k) => k !== i), "Format removed.");
  };

  const shownZones = useMemo(() => zones.filter((z) => z.toLowerCase().includes(zoneFilter.toLowerCase())).slice(0, 200), [zones, zoneFilter]);
  const canSave = !!draft.name.trim() && !!tried?.ok && !busy;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Filename formats</h1>
          <p>LanePulse reads each image's capture time from its filename. If your camera names files in a way the app does not know, show it here how the names are built.</p>
        </div>
        {editing === null && <button className="primary" onClick={() => start(-1)}>New format</button>}
      </div>
      {msg && <div className={`card msg ${msg.kind}`} role={msg.kind === "err" ? "alert" : "status"}>{msg.text}</div>}

      {editing !== null && (
        <div className="card">
          <h2>{editing === -1 ? "New format" : `Edit “${formats?.[editing]?.name}”`}</h2>

          <label className="field"><span>1. Paste one or more filenames from your camera (one per line)</span>
            <textarea rows={3} className="mono" value={examples} placeholder="2026_08_26T00_00_18_home_578695735485d7f4.jpg"
              onChange={(e) => setExamples(e.target.value)} />
          </label>
          <div className="row"><button onClick={guess} disabled={!exampleList.length}>Guess the format from the first name</button></div>

          <label className="field" style={{ marginTop: 14 }}><span>2. The format: text the names always have, and the parts that change in braces</span>
            <input ref={tplRef} className="mono" value={draft.template} placeholder="{YYYY}_{MM}_{DD}T{hh}_{mm}_{ss}_{camera}_{id}"
              onChange={(e) => setDraft((d) => ({ ...d, template: e.target.value }))} />
          </label>
          <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
            {tokens.map((t) => <button key={t.token} type="button" title={t.label} onClick={() => insert(t.token)}>
              <span className="mono">{t.token}</span> <span className="muted">{t.label}</span></button>)}
          </div>
          <p className="muted">The file extension (.jpg) is not part of the format. The camera and ID parts can be any text.</p>

          <fieldset className="field" style={{ marginTop: 14 }}>
            <legend>3. The time in the names is</legend>
            <label><input type="radio" checked={draft.zone === "UTC"} onChange={() => setDraft((d) => ({ ...d, zone: "UTC" }))} /> UTC</label>{" "}
            <label><input type="radio" checked={draft.zone !== "UTC"} onChange={() => setDraft((d) => ({ ...d, zone: d.zone === "UTC" ? "Asia/Jerusalem" : d.zone }))} /> Local time in a time zone</label>
            {draft.zone !== "UTC" && (
              <div style={{ marginTop: 6, display: "grid", gap: 6, maxWidth: 420 }}>
                <input placeholder="Search, e.g. Jerusalem" value={zoneFilter} onChange={(e) => setZoneFilter(e.target.value)} />
                <select size={4} style={{ width: "100%" }} value={draft.zone} onChange={(e) => setDraft((d) => ({ ...d, zone: e.target.value }))}>
                  {!shownZones.includes(draft.zone) && <option value={draft.zone}>{draft.zone}</option>}
                  {shownZones.map((z) => <option key={z} value={z}>{z}</option>)}
                </select>
                <p className="muted" style={{ margin: 0 }}>Local times are turned into UTC with that zone's daylight-saving rules. A time that does not exist (clocks moved forward) is not read.</p>
              </div>
            )}
          </fieldset>

          <label className="field" style={{ marginTop: 14 }}><span>4. A name for this format</span>
            <input value={draft.name} placeholder="Gate cameras" onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))} />
          </label>

          <h3>How the app reads your names</h3>
          {!draft.template.trim() && <p className="muted">Write or guess the format to see the result.</p>}
          {tried && !tried.ok && <div className="msg err">{tried.error}</div>}
          {tried?.ok && tried.results.length === 0 && <p className="muted">The format is valid ({tried.precision} precision). Paste a filename above to check it.</p>}
          {tried?.ok && tried.results.map((r, i) => (
            <div key={i} className={`msg ${r.ok ? "ok" : "err"}`} style={{ color: "inherit" }}>
              <div className="mono bidi" style={{ wordBreak: "break-all" }}>{r.name}</div>
              {r.parts && (
                <div className="row" style={{ flexWrap: "wrap", gap: "4px 14px", margin: "6px 0" }}>
                  {PART_ORDER.filter((p) => r.parts?.[p]).map((p) => (
                    <span key={p}><span className="muted">{PART_LABEL[p]}</span> <b className="mono">{r.parts![p]}</b></span>))}
                </div>
              )}
              {r.ok
                ? <div>Capture time: <b className="mono">{r.utc?.replace("T", " ").replace("Z", " UTC")}</b>{r.local && <span className="muted"> · {r.local}</span>}</div>
                : <div>Not read: {r.reason}</div>}
            </div>
          ))}

          {tried?.ok && (
            <details style={{ marginTop: 10 }}>
              <summary>Check every name in a folder</summary>
              <div className="row" style={{ marginTop: 6 }}>
                <input className="mono" style={{ flex: 1 }} placeholder={"C:\\Images\\Camera1"} value={folder} onChange={(e) => setFolder(e.target.value)} />
                <label><input type="checkbox" checked={subdirs} onChange={(e) => setSubdirs(e.target.checked)} /> Subfolders</label>
                <button onClick={checkFolder} disabled={!folder.trim() || busy}>Check</button>
              </div>
              {tried.folder?.error && <div className="msg err">{tried.folder.error}</div>}
              {tried.folder?.checked !== undefined && (
                <div className={tried.folder.matched === tried.folder.checked ? "msg ok" : "msg err"}>
                  <b>{tried.folder.matched} of {tried.folder.checked} names match.</b>
                  {!!tried.folder.not_matched_examples?.length && <div>Not matching: <span className="mono">{tried.folder.not_matched_examples.join(", ")}</span></div>}
                </div>
              )}
            </details>
          )}

          <div className="row" style={{ marginTop: 14 }}>
            <button className="primary" onClick={save} disabled={!canSave}>Save format</button>
            <button onClick={() => setEditing(null)}>Cancel</button>
            {!draft.name.trim() && draft.template.trim() && <span className="muted">Give the format a name to save it.</span>}
          </div>
        </div>
      )}

      <div className="card">
        <h2>Your formats</h2>
        {formats === null && <p className="muted">Loading…</p>}
        {formats?.length === 0 && <p className="muted">None yet.</p>}
        <p className="muted">Your formats apply to every mission and are tried before the built-in ones.</p>
        {formats?.map((f, i) => (
          <div key={i} style={{ padding: "10px 0", borderTop: i ? "1px solid var(--line, #2a313a)" : undefined }}>
            <b className="bidi">{f.name}</b> <span className="muted">· time in names: {f.zone === "UTC" ? "UTC" : `local, ${f.zone}`}</span>
            <div className="mono" style={{ wordBreak: "break-all", margin: "4px 0 6px" }}>{f.template}</div>
            <div className="row" style={{ gap: 6 }}>
              <button onClick={() => start(i)} disabled={editing !== null}>Edit</button>
              <button className="danger" onClick={() => remove(i)} disabled={busy || editing !== null}>Remove</button>
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <h2>Built-in formats</h2>
        <ul className="muted">{builtin.map((b) => <li key={b.name}><span className="mono">{b.name}</span>: {b.description}</li>)}</ul>
      </div>
    </div>
  );
}
