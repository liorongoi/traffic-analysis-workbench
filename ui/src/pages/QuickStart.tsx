import { useEffect, useState } from "react";
import { api, ApiError, post } from "../api";
import { go } from "../router";
import FolderPicker from "../components/FolderPicker";
import { NamePick } from "../components/Pick";
import { DETECTOR_LABEL, GradeLine, partsOf, toggleDetector, type Grades } from "./Wizard";

// A new mission on one screen. The person chooses the folder; LanePulse reads it and fills in the rest (name, still
// images or video, the time zone the last mission used, an exports folder beside the source), then Create and start
// creates the mission and starts processing in one press. Every guess can be changed first; the full step-by-step
// setup stays one link away for anything else.

type Ref = { countries: Record<string, string>; modes: Record<string, string[]>; zones: string[];
  sam3?: { installed: boolean; reason: string | null }; rfdetr?: { installed: boolean; reason: string | null } };
type Guess = { source_dir: string; name: string; input_type: string; images: number; videos: number; readable: number; more: boolean;
  pattern: string | null; first_utc: string | null; last_utc: string | null; unreadable: { file: string; reason: string }[];
  iana_zone: string; country: string; zone_from: string | null; output_dir: string; output_exists: boolean };

const MODE_TEXT: Record<string, string> = {
  general_traffic: "General traffic: vehicles in each image",
  object_search: "Object search: find vehicles that match a description",
  plate_recognition: "Plate recognition: read licence plates",
  unique_targets: "Unique vehicles: each vehicle counted once",
  indoor_outdoor: "Sort into indoor and outdoor: two groups of images",
};
const modeText = (m: string) => MODE_TEXT[m] ?? m.replace(/_/g, " ");

function span(a: string, b: string) {
  const ms = new Date(b).valueOf() - new Date(a).valueOf();
  const h = ms / 3600000;
  if (h < 1) return `${Math.max(1, Math.round(ms / 60000))} min`;
  if (h < 48) return `${Math.round(h)} h`;
  return `${Math.round(h / 24)} days`;
}
const when = (iso: string) => new Date(iso).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" });

/** Names to pick from: the one read from the folder first, then shorter ones with the month. */
function nameIdeas(g: Guess): string[] {
  const place = g.name.split(", ")[0];
  const month = g.first_utc ? new Date(g.first_utc).toLocaleDateString("en-GB", { month: "long", year: "numeric", timeZone: "UTC" }) : null;
  return [g.name, ...(month ? [`${place}, ${month}`, `Traffic, ${month}`] : [place]), `${place}, ${g.input_type === "video" ? "video" : "images"}`];
}

export default function QuickStart() {
  const [ref, setRef] = useState<Ref | null>(null);
  const [folder, setFolder] = useState<string[]>([]);
  const [g, setG] = useState<Guess | null>(null);
  const [reading, setReading] = useState(false);
  const [readErr, setReadErr] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [zone, setZone] = useState("");
  const [mode, setMode] = useState("");
  const [out, setOut] = useState<string[]>([]);
  const [changeOut, setChangeOut] = useState(false);
  const [detector, setDetector] = useState("standard");
  const [watch, setWatch] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string[]>([]);

  useEffect(() => { api("/reference").then(setRef).catch(() => setRef(null)); }, []);
  const grades: Grades | undefined = (ref as any)?.grades?.scenes ? (ref as any).grades : undefined;
  // Read the folder once the path stops changing (the picker reports every key typed in its field).
  useEffect(() => {
    const dir = folder[0]?.trim();
    setReadErr(null);
    if (!dir) { setG(null); return; }
    let live = true;
    const t = window.setTimeout(async () => {
      setReading(true);
      try {
        const r: Guess = await post("/quick-start", { source_dir: dir, include_subdirs: true });
        if (!live) return;
        setG(r); setName(r.name); setZone(r.iana_zone); setOut([r.output_dir]); setChangeOut(false); setErr([]);
        setMode(ref?.modes[r.input_type]?.[0] ?? "general_traffic");
      } catch (e) {
        if (live) { setG(null); setReadErr((e as ApiError).message); }
      } finally { if (live) setReading(false); }
    }, 450);
    return () => { live = false; window.clearTimeout(t); };
  }, [folder[0], ref]);

  const files = g ? g.images + g.videos : 0;
  const video = g?.input_type === "video";
  const samOk = !!ref?.sam3?.installed;
  const rfOk = !!ref?.rfdetr?.installed;
  const sorting = mode === "indoor_outdoor";      // needs no capture time and no detector

  const create = async () => {
    if (!g) return;
    setBusy(true); setErr([]);
    try {
      const output = out[0]?.trim() || g.output_dir;
      if (output === g.output_dir && !g.output_exists) await post("/quick-start/output", { path: output });
      const body = { name: name.trim() || g.name, input_type: g.input_type, mode, country: g.country, iana_zone: zone || g.iana_zone,
        source_dir: g.source_dir, include_subdirs: true, processing: watch ? "continuous" : "one_time", output_dir: output,
        export_formats: ["xlsx", "csv"], detector };
      const check = await post("/missions/validate", body);
      if (check.errors.length) { setErr(check.errors); return; }
      const r = await post("/missions", body);
      await post(`/missions/${r.id}/job`);
      if (watch) await api(`/missions/${r.id}/monitoring`, { method: "PUT", body: { enabled: true } }).catch(() => undefined);
      go(mode === "indoor_outdoor" ? `/m/${r.id}/sort` : `/m/${r.id}`);
    } catch (e) {
      setErr([(e as ApiError).message]);
    } finally { setBusy(false); }
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>New mission</h1>
          <p>Choose the folder with the camera's images or recordings. LanePulse reads it and fills in the rest. Your files are never changed.</p>
        </div>
        <a href="#/new/full">Step-by-step setup with every option</a>
      </div>
      <div className="qs">
        <div style={{ display: "grid", gap: 16 }}>
          <div className="card">
            <FolderPicker label="Source folder" value={folder} onChange={setFolder} />
            {reading && <p className="muted" role="status" aria-live="polite"><span className="pulse" />Reading the folder…</p>}
            {readErr && !reading && folder[0]?.trim() && <p className="muted" role="status">{readErr}</p>}
            {g && !reading && (
              <>
                <div className="qs-figs" style={{ marginTop: 4 }}>
                  <div><b>{files.toLocaleString()}{g.more ? "+" : ""}</b><span>{video ? (g.images ? `${g.videos} videos, ${g.images} images` : "Video recordings") : "Still images"}</span></div>
                  <div><b>{g.readable.toLocaleString()}</b><span>Capture times read</span></div>
                  <div><b>{g.first_utc && g.last_utc ? span(g.first_utc, g.last_utc) : "–"}</b><span>{g.first_utc && g.last_utc ? `${when(g.first_utc)} to ${when(g.last_utc)} UTC` : "No capture times"}</span></div>
                </div>
              </>
            )}
          </div>
          {g && !reading && files === 0 && <div className="msg warn">There are no images or videos in this folder or its subfolders.</div>}
          {g && !reading && files > 0 && g.readable === files && g.pattern &&
            <div className="msg ok">Every file name gives its capture time, so each {video ? "recording" : "image"} is placed in time to the second.</div>}
          {g && !reading && files > 0 && g.readable < files && !sorting && (
            <div className="msg warn">
              {files - g.readable} of {files} file names have no capture time LanePulse can read{g.unreadable[0] ? <>, for example <span className="mono bidi">{g.unreadable[0].file}</span></> : null}. Those files would be skipped.{" "}
              <a href="#/formats" target="_blank" rel="noreferrer">Show the app how this camera names its files</a> (opens in a new tab), then choose the folder again.
            </div>
          )}
        </div>

        <div className="card" aria-disabled={!g}>
          <h2>{g ? "What LanePulse filled in" : "Choose a folder to start"}</h2>
          <p className="muted" style={{ marginTop: -6 }}>{g ? "Change anything before you start." : "The name, time zone and exports folder are filled in from the folder you choose."}</p>
          <div className="guess"><span>Name</span>
            <span>{g ? <NamePick label="Mission name" value={name} onChange={setName} suggestions={nameIdeas(g)} style={{ width: "100%" }} />
                : <select disabled aria-label="Mission name"><option>Chosen from the folder</option></select>}
              <span className="why">{g ? "From the folder and the dates in it" : " "}</span></span><span /></div>
          <div className="guess"><span>Time zone</span>
            <span><select value={zone} disabled={!g || !ref} onChange={(e) => setZone(e.target.value)} aria-label="Time zone">
                {zone && !ref?.zones.includes(zone) && <option value={zone}>{zone}</option>}
                {(ref?.zones ?? []).map((z) => <option key={z} value={z}>{z}</option>)}
              </select>
              <span className="why">{g?.zone_from ? `The last mission, ${g.zone_from}, used it` : g ? "The usual zone for this workstation" : " "}</span></span><span /></div>
          <div className="guess"><span>What to find</span>
            <span><select value={mode} disabled={!g || !ref} onChange={(e) => setMode(e.target.value)} aria-label="What to find">
                {(ref && g ? ref.modes[g.input_type] ?? [] : []).map((m) => <option key={m} value={m}>{modeText(m)}</option>)}
              </select>
              <span className="why">{g ? (sorting ? "Two groups, indoor and outdoor; it cannot change later" : `The usual choice for ${video ? "video" : "still images"}; it cannot change later`) : " "}</span></span><span /></div>
          <div className="guess"><span>Exports</span>
            <span>{changeOut
              ? <FolderPicker label="Exports folder" value={out} onChange={setOut} placeholder="Folder for Excel and CSV files" />
              : <><span className="mono bidi" style={{ overflowWrap: "anywhere" }}>{out[0] || "–"}</span>
                <span className="why">{g ? `Excel and CSV, ${g.output_exists ? "in this folder" : "in a new folder beside the source"}` : " "}</span></>}</span>
            {g && !changeOut ? <button className="linkish" onClick={() => setChangeOut(true)}>Change</button> : <span />}</div>

          {sorting ? <p className="muted" style={{ marginTop: 16 }}>Each image is placed in indoor or outdoor. You can move any of them afterwards, and LanePulse learns from your moves. File names need no capture time.</p> :
          <div style={{ marginTop: 16 }}>
            <div>
              <div className="muted" style={{ fontSize: 13, marginBottom: 8 }}>Detector</div>
              <div className="muted" style={{ fontSize: 12.5, margin: "-4px 0 8px" }}>Tick one fast detector, SAM 3.1, or a fast one with SAM 3.1 (their findings are merged).</div>
              <div className="seg" role="group" aria-label="Detector">
                {([["standard", "YOLOX-s", "Fastest, works on any computer"], ["rfdetr", "RF-DETR", rfOk ? "Finds more, a little slower" : "Not installed here"],
                  ["sam3", "SAM 3.1", samOk ? "Finds things by name, slowest" : "Not installed here"]] as const).map(([k, l, d]) => {
                  const on = k === "sam3" ? partsOf(detector).sam : partsOf(detector).fast === k;
                  return <button key={k} type="button" role="checkbox" aria-checked={on} className={on ? "on" : ""}
                    disabled={!g || (k === "rfdetr" ? !rfOk : k === "sam3" && !samOk)} onClick={() => setDetector(toggleDetector(detector, k))}>
                    <b><span aria-hidden="true">{on ? "\u2611" : "\u2610"}</span> {l}</b><small>{d}</small><GradeLine g={grades} k={k} /></button>;
                })}
              </div>
              <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>Runs: {DETECTOR_LABEL[detector] ?? detector}</div>
            </div>
          </div>}

          {err.map((e) => <div key={e} className="msg err" role="alert">{e}</div>)}
          <div className="qs-go">
            <label className="row" style={{ color: "var(--text-2)" }}><input type="checkbox" checked={watch} disabled={!g} onChange={(e) => setWatch(e.target.checked)} /> Keep watching the folder for new files</label>
            <span className="grow" />
            <button className="primary big" disabled={!g || busy || files === 0 || (g.readable === 0 && !sorting) || !mode} onClick={create}>{busy ? "Creating…" : "Create and start"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
