import { useEffect, useState } from "react";
import { api, ApiError, post } from "../api";
import { go } from "../router";
import FolderPicker from "../components/FolderPicker";
import { NamePick } from "../components/Pick";
import { CameraChoice, DETECTOR_LABEL, DetectorChoice, usesSam } from "./Wizard";
import { fmtDuration } from "../api";
import { MODE_TEXT as SET_MODE } from "./Settings";
import { TargetsEditor } from "../components/Targets";

// A new mission is its input folder. The person chooses the folder; LanePulse reads it, names the mission, and takes
// everything decided before analysis (what to find, the detector, the results folder, the time zone) from Settings.
// Start creates the mission and starts processing in one press. A new kind of file name is learned here, once, and
// remembered for every mission after.

type Guess = { source_dir: string; name: string; input_type: string; images: number; videos: number; readable: number; more: boolean;
  pattern: string | null; first_utc: string | null; last_utc: string | null; unreadable: { file: string; reason: string }[];
  iana_zone: string; country: string; zone_from: string | null; output_dir: string; output_exists: boolean;
  settings: Record<string, any> };
type Learned = { template: string | null; total: number; matched: number; examples: { name: string; utc: string }[]; reason?: string; saved?: boolean; name?: string };

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
  const [folder, setFolder] = useState<string[]>([]);
  const [g, setG] = useState<Guess | null>(null);
  const [reading, setReading] = useState(false);
  const [readErr, setReadErr] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string[]>([]);
  const [learn, setLearn] = useState<Learned | null>(null);
  const [again, setAgain] = useState(0);
  // Video: the files themselves, and this mission's own choices when they differ from Settings.
  const [vids, setVids] = useState<{ videos: any[]; more: boolean } | null>(null);
  const [own, setOwn] = useState<Record<string, any>>({});
  const [changing, setChanging] = useState(false);
  const [sam, setSam] = useState<any>(null);
  useEffect(() => { api("/reference").then((r) => setSam(r.sam3 ?? null)).catch(() => undefined); }, []);
  const [words, setWords] = useState<any>(null);   // the default words per vehicle class, for "What to look for"
  useEffect(() => { api("/sam3").then(setWords).catch(() => undefined); }, []);

  // Read the folder once the path stops changing (the picker reports every key typed in its field).
  useEffect(() => {
    const dir = folder[0]?.trim();
    setReadErr(null); setLearn(null);
    if (!dir) { setG(null); return; }
    let live = true;
    const t = window.setTimeout(async () => {
      setReading(true);
      try {
        const r: Guess = await post("/quick-start", { source_dir: dir, include_subdirs: true });
        if (!live) return;
        setG(r); setName((n) => (again && n ? n : r.name)); setErr([]);
        setVids(null);
        if (r.input_type === "video") post("/quick-start/videos", { source_dir: r.source_dir, include_subdirs: true })
          .then((v) => live && setVids(v)).catch(() => undefined);
        // Names this app cannot read yet: guess their format straight away, so one press teaches it.
        if (r.readable < r.images + r.videos) {
          post("/name-formats/learn", { folder: r.source_dir, zone: "UTC" }).then((l) => live && setLearn(l)).catch(() => undefined);
        }
      } catch (e) {
        if (live) { setG(null); setReadErr((e as ApiError).message); }
      } finally { if (live) setReading(false); }
    }, 450);
    return () => { live = false; window.clearTimeout(t); };
  }, [folder[0], again]);

  const files = g ? g.images + g.videos : 0;
  const video = g?.input_type === "video";
  const st = { ...(g?.settings ?? {}), ...own };

  const remember = async () => {
    if (!g) return;
    try { setLearn(await post("/name-formats/learn", { folder: g.source_dir, zone: "UTC", save: true })); setAgain((n) => n + 1); }
    catch (e) { setErr([(e as ApiError).message]); }
  };

  // mode: this mission only, in place of the mode in Settings (sorting a bundle into indoor and outdoor).
  const create = async (mode?: string) => {
    if (!g) return;
    setBusy(true); setErr([]);
    try {
      if (!g.output_exists) await post("/quick-start/output", { path: g.output_dir });
      const body = { ...st, name: name.trim() || g.name, input_type: g.input_type, country: g.country, iana_zone: g.iana_zone,
        source_dir: g.source_dir, include_subdirs: true, output_dir: g.output_dir, ...(mode ? { mode } : {}) };
      const check = await post("/missions/validate", body);
      if (check.errors.length) { setErr(check.errors); return; }
      const r = await post("/missions", body);
      await post(`/missions/${r.id}/job`);
      if (st.processing === "continuous") await api(`/missions/${r.id}/monitoring`, { method: "PUT", body: { enabled: true } }).catch(() => undefined);
      go(body.mode === "indoor_outdoor" ? `/m/${r.id}/sort` : `/m/${r.id}`);
    } catch (e) {
      setErr([(e as ApiError).message]);
    } finally { setBusy(false); }
  };

  // Videos need no date or time in their names: they are analysed by their own clock, and a start time can be set
  // for each one later on the mission's video page.
  const ready = !!g && !reading && files > 0 && (video || g.readable > 0);
  return (
    <div className="page newm">
      <div className="newm-hero">
        <h1>New mission</h1>
        <p>Choose the input folder. Everything else comes from <a href={`#/settings/${video ? "video" : g ? "images" : "both"}`}>Settings</a>.</p>
        <div className="newm-pick"><FolderPicker label="Input folder" value={folder} onChange={setFolder} placeholder="Type a path or press Browse" /></div>
        {reading && <p className="muted" role="status" aria-live="polite"><span className="pulse" />Reading the folder…</p>}
        {readErr && !reading && folder[0]?.trim() && <p className="msg err" role="status">{readErr}</p>}
      </div>

      {g && !reading && (
        <div className="newm-card card">
          {!video && <div className="newm-figs">
            <div><b>{files.toLocaleString()}{g.more ? "+" : ""}</b><span>still images</span></div>
            <div><b>{g.readable.toLocaleString()}</b><span>capture times read</span></div>
            <div><b>{g.first_utc && g.last_utc ? span(g.first_utc, g.last_utc) : "–"}</b><span>{g.first_utc && g.last_utc ? `${when(g.first_utc)} to ${when(g.last_utc)} UTC` : "no capture times"}</span></div>
          </div>}
          {video && <div className="newm-vids">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>{g.videos.toLocaleString()}{g.more ? "+" : ""} video{g.videos === 1 ? "" : "s"}{vids ? ` · ${fmtDuration(vids.videos.reduce((a, v) => a + (v.duration_s ?? 0), 0))}${vids.more ? "+" : ""} in all` : ""}</b>
              <span className="muted small">{g.readable ? `${g.readable} with a start time in the name; the rest are timed from their own start` : "Timed from the start of each video; no date is needed in the names"}</span>
            </div>
            {!vids && <p className="muted small" role="status">Opening the videos…</p>}
            {vids && <div className="newm-vlist">{vids.videos.map((v) => (
              <div key={v.rel_path} className={`newm-v${v.error ? " bad" : ""}`}>
                {v.thumb ? <img src={v.thumb} alt="" /> : <div className="ph" />}
                <span className="mono bidi" title={v.rel_path}>{v.rel_path}</span>
                <span className="muted small">{v.error ? "cannot be opened" : `${fmtDuration(v.duration_s)}${v.width ? ` · ${v.width}×${v.height}` : ""}${v.codec ? ` · ${v.codec}` : ""}`}</span>
              </div>))}
              {vids.more && <div className="muted small">and more…</div>}
            </div>}
            {vids && vids.videos.some((v) => v.error) && <div className="msg warn">{vids.videos.filter((v) => v.error).length} file(s) cannot be opened and will be listed as such: {vids.videos.filter((v) => v.error).map((v) => v.rel_path).join(", ")}.</div>}
          </div>}
          {files === 0 && <div className="msg warn">There are no images or videos in this folder or its subfolders.</div>}
          {files > 0 && !video && g.readable < files && (
            <div className="msg warn learn">
              <div>{files - g.readable} of {files} file names have no capture time LanePulse knows yet{g.unreadable[0] ? <>, for example <span className="mono bidi">{g.unreadable[0].file}</span></> : null}.</div>
              {learn?.template && !learn.saved && <div className="row" style={{ marginTop: 8 }}>
                <span>They look like <code className="mono">{learn.template}</code>: {learn.matched} of {learn.total} read this way{learn.examples[0] ? <>, for example {when(learn.examples[0].utc)} UTC</> : null}.</span>
                <button className="primary" onClick={remember}>Remember this format</button>
                <a href="#/settings/formats">Teach it by hand</a></div>}
              {learn && !learn.template && <div style={{ marginTop: 6 }}>{learn.reason ?? "No date and time could be found in these names."} <a href="#/settings/formats">Teach the format by hand</a>.</div>}
            </div>)}
          {learn?.saved && <div className="msg ok">Format remembered as “{learn.name ?? "learned"}”. Every mission reads such names from now on.</div>}

          <div className="newm-name">
            <span>Name</span>
            <NamePick label="Mission name" value={name} onChange={setName} suggestions={nameIdeas(g)} style={{ width: "100%" }} />
          </div>
          <div className="newm-uses">
            <span className="muted">{Object.keys(own).length ? "Settings, changed for this mission" : "From Settings"}</span>
            <span className="chip">{video && st.mode === "general_traffic" ? "Count and track vehicles" : SET_MODE[st.mode] ?? modeText(st.mode ?? "")}</span>
            <span className="chip">{DETECTOR_LABEL[st.detector] ?? st.detector}</span>
            {video && <span className="chip">{st.camera === "moving" ? "Moving camera" : "Fixed camera"}</span>}
            {!video && <span className="chip">{g.iana_zone}</span>}
            <span className="chip">{video ? (st.analyse_new ? "New videos start by themselves" : "New videos wait for you") : st.processing === "continuous" ? "Keeps watching" : "Analyse once"}</span>
            <span className="chip" title={g.output_dir}>Results: <span className="mono bidi">{g.output_dir.split(/[\\/]/).slice(-2).join("/")}</span></span>
            {video ? <button className="linkish" aria-expanded={changing} onClick={() => setChanging(!changing)}>{changing ? "Done" : "Change for this mission"}</button>
              : <a href={`#/settings/${video ? "video" : "images"}`}>Change</a>}
          </div>
          {/* What to look for, from Settings, changed here for this mission only (pedestrians, tractors, several at once). */}
          {words?.vehicle_prompts && st.mode !== "indoor_outdoor" && <div className="newm-targets">
            <TargetsEditor title="Looks for" words={st.sam3_words ?? null} uncounted={st.sam3_uncounted ?? null} defaults={words.vehicle_prompts}
              busy={busy} samOnly={st.detector === "sam3"} standardOnly={!usesSam(st.detector ?? "standard")} standardKnows={words.standard_knows ?? []}
              onSave={async (w, u) => setOwn({ ...own, sam3_words: w, sam3_uncounted: u.length ? u : null })} />
          </div>}
          {video && changing && <div className="newm-own">
            <div className="field"><span>What to find</span>
              <div className="seg" role="radiogroup" aria-label="What to find">
                {[["general_traffic", "Count and track vehicles"], ["object_search", "Find vehicles by description"], ["plate_recognition", "Read licence plates"]].map(([k, l]) => (
                  <button key={k} role="radio" aria-checked={st.mode === k} className={st.mode === k ? "on" : ""} onClick={() => setOwn({ ...own, mode: k })}><b>{l}</b></button>))}
              </div></div>
            <DetectorChoice value={st.detector ?? "standard"} onChange={(d) => setOwn({ ...own, detector: d })} sam={sam ?? undefined} video />
            <CameraChoice value={st.camera ?? "fixed"} onChange={(c) => setOwn({ ...own, camera: c })} />
            <div className="row"><span className="muted small">These apply to this mission only. Settings stay as they are.</span>
              {Object.keys(own).length > 0 && <button className="linkish" onClick={() => setOwn({})}>Back to Settings</button>}</div>
          </div>}
          {err.map((e) => <div key={e} className="msg err" role="alert">{e}</div>)}
          <div className="newm-go">
            {!video && <a href="#/new/full" className="muted">Different choices for this mission only</a>}
            <span className="grow" />
            {!video && <button className="big" disabled={!g || reading || files === 0 || busy} onClick={() => create("indoor_outdoor")}
              title="Split these images into indoor and outdoor instead; file names need no capture time">Sort into indoor and outdoor</button>}
            <button className="primary big" disabled={!ready || busy} onClick={() => create()}>{busy ? "Starting…" : video && g && g.videos > 0 ? `Start analysing ${g.videos.toLocaleString()}${g.more ? "+" : ""} video${g.videos === 1 ? "" : "s"}` : "Start"}</button>
          </div>
        </div>
      )}
    </div>
  );
}
