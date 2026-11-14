import { useEffect, useState } from "react";
import { api, ApiError, post } from "../api";
import FolderPicker from "../components/FolderPicker";
import { ZonePick } from "../components/Pick";
import { MapServers } from "../components/MapSources";
import { CameraChoice, DetectorChoice, ModelChoice, ParallelChoice, SamInterval, usesSam, partsOf } from "./Wizard";
import NameFormats from "./NameFormats";
import Workers from "./Workers";

// Settings: everything decided once, before analysis, so the working pages show only what is done during and after a
// run. In three parts, as the operator thinks of them: for still images and video alike, for still images only, and
// for video only. New missions take these; an existing mission's own setup is behind the cog on its page.
type Defaults = {
  both: { results_folder: string | null; export_formats: string[]; iana_zone: string | null; country: string | null; watch: boolean;
    run_mode: string; teacher: string; auto_min_reviews: number; auto_epochs: number };
  images: { detector: string; mode: string; vehicle_model: string | null };
  video: { detector: string; mode: string; sam3_interval_s: number; camera: string; video_workers: number; analyse_new: boolean };
};
type Part = keyof Defaults;

export const MODE_TEXT: Record<string, string> = {
  general_traffic: "Count traffic: every vehicle in each image",
  object_search: "Find vehicles that match a description",
  plate_recognition: "Read licence plates",
};

const PARTS: [string, string, string][] = [
  ["both", "Both", "For still images and video alike"],
  ["images", "Still images", "Only for missions of still images"],
  ["video", "Video", "Only for video missions"],
];
// Older addresses of what is now a part of Settings.
const LEGACY: Record<string, string> = { computer: "both", formats: "both", workers: "video", maps: "video" };

export default function Settings({ section }: { section?: string }) {
  const part = LEGACY[section ?? ""] ?? (PARTS.some(([k]) => k === section) ? section! : "both");
  const [d, setD] = useState<Defaults | null>(null);
  const [ref, setRef] = useState<any>(null);
  const [sam, setSam] = useState<any>(null);
  const [saved, setSaved] = useState<any[]>([]);
  const [note, setNote] = useState<{ kind: string; text: string } | null>(null);
  useEffect(() => {
    api("/settings/missions").then(setD).catch((e) => setNote({ kind: "err", text: (e as ApiError).message }));
    api("/reference").then(setRef).catch(() => setRef(null));
    api("/sam3").then(setSam).catch(() => setSam({ installed: false, reason: "Could not check." }));
    api("/models/saved").then((r) => setSaved(r.saved ?? [])).catch(() => setSaved([]));
  }, []);
  // Each change is kept at once: there is no Save button to forget.
  const change = async <P extends Part>(p: P, values: Partial<Defaults[P]>) => {
    if (!d) return;
    const before = d;
    setD({ ...d, [p]: { ...d[p], ...values } });
    try { setD(await api("/settings/missions", { method: "PUT", body: { [p]: values } })); setNote({ kind: "ok", text: "Saved. New missions use this." }); }
    catch (e) { setD(before); setNote({ kind: "err", text: (e as ApiError).message }); }
  };
  useEffect(() => { if (note?.kind !== "ok") return; const t = window.setTimeout(() => setNote(null), 2500); return () => window.clearTimeout(t); }, [note]);
  useEffect(() => {   // #/settings/formats and the like land on their card
    const id = section && LEGACY[section] ? `set-${section}` : null;
    if (id) window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ block: "start" }), 300);
  }, [section]);

  return (
    <div className="page settings">
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <p>What you decide once, before analysis. New missions take these, so a new mission only needs its input folder.</p>
        </div>
        {note && <span className={`msg ${note.kind} settings-note`} role={note.kind === "err" ? "alert" : "status"}>{note.text}</span>}
      </div>
      <nav className="tabs settings-parts" aria-label="Settings parts">
        {PARTS.map(([k, l, hint]) => <a key={k} href={`#/settings/${k}`} className={part === k ? "on" : ""} aria-current={part === k ? "page" : undefined}>
          {l}<small>{hint}</small></a>)}
      </nav>
      {!d ? <p className="muted">Loading…</p> : <>
        {part === "both" && <div className="set-grid">
          <Card title="Results" id="set-results" lead="Where each mission writes its Excel and CSV files.">
            <div className="seg two" role="radiogroup" aria-label="Results folder">
              <button role="radio" aria-checked={!d.both.results_folder} className={!d.both.results_folder ? "on" : ""}
                onClick={() => change("both", { results_folder: null })}><b>Beside each input folder</b><small>A new folder “… LanePulse exports” next to it</small></button>
              <button role="radio" aria-checked={!!d.both.results_folder} className={d.both.results_folder ? "on" : ""}
                onClick={() => { if (!d.both.results_folder) setD({ ...d, both: { ...d.both, results_folder: "" } }); }}><b>In one results folder</b><small>Each mission gets its own folder inside it</small></button>
            </div>
            {d.both.results_folder !== null && <FolderPicker label="Results folder" value={d.both.results_folder ? [d.both.results_folder] : []}
              onChange={(v) => v[0]?.trim() && v[0] !== d.both.results_folder && change("both", { results_folder: v[0] })} placeholder="Choose the results folder" />}
            <div className="row" role="group" aria-label="File types" style={{ marginTop: 10 }}>
              <span className="muted">File types</span>
              {[["xlsx", "Excel"], ["csv", "CSV"]].map(([k, l]) => <label key={k} className="row"><input type="checkbox" checked={d.both.export_formats.includes(k)}
                onChange={(e) => change("both", { export_formats: e.target.checked ? [...d.both.export_formats, k] : d.both.export_formats.filter((x) => x !== k) })} />{l}</label>)}
            </div>
          </Card>
          <Card title="Time zone" id="set-zone" lead="Capture times are kept in UTC and shown in this zone. The country decides which plates can be read.">
            <label className="field"><span>Time zone of the cameras</span>
              <div className="row">
                <select aria-label="Time zone source" value={d.both.iana_zone ? "fixed" : "last"} onChange={(e) => change("both", { iana_zone: e.target.value === "last" ? null : (ref?.country_zones?.IL ?? "Asia/Jerusalem") })}>
                  <option value="last">The same as the latest mission</option>
                  <option value="fixed">Always this zone</option>
                </select>
                {d.both.iana_zone && ref && <ZonePick zones={ref.zones} value={d.both.iana_zone} onChange={(z) => change("both", { iana_zone: z })} />}
              </div></label>
            {d.both.iana_zone && ref && <label className="field"><span>Country</span>
              <select value={d.both.country ?? ""} onChange={(e) => change("both", { country: e.target.value || null })}>
                <option value="">From the time zone</option>
                {Object.entries(ref.countries as Record<string, string>).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select></label>}
          </Card>
          <Card title="New files" id="set-watch" lead="Whether a mission keeps watching its input folder after the first run. Start or stop it on each mission's page at any time.">
            <div className="seg two" role="radiogroup" aria-label="New files">
              {([[false, "Analyse once", "The files there when the mission starts"], [true, "Keep watching", "New files are analysed as they arrive"]] as const).map(([v, l, h]) =>
                <button key={l} role="radio" aria-checked={d.both.watch === v} className={d.both.watch === v ? "on" : ""} onClick={() => change("both", { watch: v })}><b>{l}</b><small>{h}</small></button>)}
            </div>
          </Card>
          <Card title="Learning" id="set-learning" lead="How a mission's model improves from your reviews.">
            <div className="seg two" role="radiogroup" aria-label="Run mode">
              {([["manual", "I decide each step", "Freeze data, train and compare when you choose"], ["auto", "Full auto", "Trains a round after enough new reviews, keeps the better model"]] as const).map(([v, l, h]) =>
                <button key={v} role="radio" aria-checked={d.both.run_mode === v} className={d.both.run_mode === v ? "on" : ""} onClick={() => change("both", { run_mode: v })}><b>{l}</b><small>{h}</small></button>)}
            </div>
            {d.both.run_mode === "auto" && <div className="row" style={{ marginTop: 10 }}>
              <label className="row">A round after
                <select value={d.both.auto_min_reviews} onChange={(e) => change("both", { auto_min_reviews: Number(e.target.value) })}>
                  {[5, 10, 20, 50, 100, 200].map((n) => <option key={n} value={n}>{n} reviews</option>)}</select></label>
              <label className="row">Epochs
                <select value={d.both.auto_epochs} onChange={(e) => change("both", { auto_epochs: Number(e.target.value) })}>
                  {[10, 20, 30, 50, 100].map((n) => <option key={n} value={n}>{n}</option>)}</select></label>
              <label className="row">Learn from
                <select value={d.both.teacher} onChange={(e) => change("both", { teacher: e.target.value })}>
                  <option value="none">My reviews only</option>
                  <option value="sam3" disabled={!sam?.installed}>My reviews and SAM 3.1</option></select></label>
            </div>}
          </Card>
          <div className="set-wide" id="set-formats"><NameFormats embedded /></div>
          <div className="set-wide" id="set-computer"><ThisComputer /></div>
        </div>}

        {part === "images" && <div className="set-grid">
          <Card title="What to find" lead="The kind of analysis a new mission of still images runs.">
            <Modes kind="images" value={d.images.mode} onChange={(v) => change("images", { mode: v })} />
          </Card>
          <Card title="Model" lead="A model trained here and saved by name, instead of the one in production use.">
            {d.images.detector !== "sam3" ? <ModelChoice saved={saved} value={d.images.vehicle_model} onChange={(v) => change("images", { vehicle_model: v })} rfdetr={partsOf(d.images.detector).fast === "rfdetr"} />
              : <p className="muted">SAM 3.1 alone has no model to choose.</p>}
          </Card>
          <Card title="Detector" wide lead="Which detectors look at each image.">
            <DetectorChoice value={d.images.detector} onChange={(v) => change("images", { detector: v })} sam={sam ?? undefined} legend={false} />
          </Card>
        </div>}

        {part === "video" && <div className="set-grid">
          <Card title="What to find" lead="The kind of analysis a new video mission runs.">
            <Modes kind="video" value={d.video.mode} onChange={(v) => change("video", { mode: v })} />
          </Card>
          <Card title="Camera and speed" lead="How recordings are analysed.">
            <CameraChoice value={d.video.camera} onChange={(v) => change("video", { camera: v })} />
            <ParallelChoice value={d.video.video_workers} onChange={(v) => change("video", { video_workers: v })} />
            {usesSam(d.video.detector) && <SamInterval value={d.video.sam3_interval_s} onChange={(v) => change("video", { sam3_interval_s: v })} both={partsOf(d.video.detector).fast !== null} />}
            <label className="field"><span>New videos in the input folder</span>
              <select value={d.video.analyse_new ? "auto" : "wait"} onChange={(e) => change("video", { analyse_new: e.target.value === "auto" })}>
                <option value="wait">Wait until I tick them in the catalog</option>
                <option value="auto">Analyse them as they arrive</option>
              </select></label>
          </Card>
          <Card title="Detector" wide lead="Which detectors look at each analysed frame.">
            <DetectorChoice value={d.video.detector} onChange={(v) => change("video", { detector: v })} sam={sam ?? undefined} legend={false} video />
          </Card>
          <div className="set-wide" id="set-workers"><Workers embedded /></div>
          <Card title="Map and ground heights" wide id="set-maps" lead="Our own map and height servers, for putting a camera on the map. Set up once; then picked from a list.">
            <MapServers />
          </Card>
        </div>}
      </>}
    </div>
  );
}

function Card({ title, lead, wide, id, children }: { title: string; lead?: string; wide?: boolean; id?: string; children: React.ReactNode }) {
  return (
    <section className={`card${wide ? " set-wide" : ""}`} id={id}>
      <h2 style={{ marginTop: 0 }}>{title}</h2>
      {lead && <p className="muted" style={{ marginTop: -4 }}>{lead}</p>}
      {children}
    </section>
  );
}

function Modes({ kind, value, onChange }: { kind: "images" | "video"; value: string; onChange: (v: string) => void }) {
  const list = kind === "video" ? ["object_search", "plate_recognition"] : ["general_traffic", "object_search", "plate_recognition"];
  return (
    <div className="seg col" role="radiogroup" aria-label="What to find">
      {list.map((m) => <button key={m} role="radio" aria-checked={value === m} className={value === m ? "on" : ""} onClick={() => onChange(m)}>
        <b>{MODE_TEXT[m]}</b></button>)}
    </div>
  );
}

function ThisComputer() {
  const [sys, setSys] = useState<any>(null);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const load = () => api("/system").then(setSys).catch((e) => setErr(e instanceof Error ? e.message : "Could not load system status."));
  useEffect(() => { load(); }, []);
  const backupNow = async () => {
    setMsg(null);
    try { const r = await post("/admin/backup"); setMsg({ kind: "ok", text: `Backed up to ${r.folder}` }); }
    catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
    load();
  };
  return (
    <div className="grid cols-2" style={{ gap: 14, alignItems: "start" }}>
      {err && <div className="card msg err" role="alert"><b>Could not load system status.</b><p>{err}</p><button onClick={() => { setErr(""); load(); }}>Try again</button></div>}
      {sys && <System sys={sys} msg={msg} backupNow={backupNow} />}
      <SizesCard />
    </div>
  );
}

/** Which size of each fast detector runs here: chosen from this computer's hardware, or by the operator. */
function SizesCard() {
  const [h, setH] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api("/hardware").then(setH).catch((e) => setErr(String(e?.message ?? e))); }, []);
  const choose = (family: string, size: string) =>
    api("/hardware/sizes", { method: "PUT", body: { family, size: size || null } }).then((r) => { setH(r); setErr(null); })
      .catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  if (!h) return err ? <div className="card"><div className="msg err">{err}</div></div> : null;
  const hw = h.hardware;
  const gpus = (hw.nvidia ?? []).map((g: any) => `${g.name} (${g.memory_gb} GB)`).join(", ");
  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Detector sizes on this computer</h2>
      <p style={{ marginTop: 0 }}>This computer: {hw.cpu_threads} processor threads, {hw.memory_gb ?? "?"} GB memory
        {gpus ? `, ${gpus}` : hw.apple_silicon ? ", Apple GPU" : ", no NVIDIA GPU"}. Detection runs on <b>{h.run_on}</b>;
        training on <b>{h.train_on}</b>. Larger sizes find more but are slower.</p>
      <table className="t">
        <thead><tr><th>Detector</th><th>Size for detection</th><th>Recommended for training</th><th>Installed</th></tr></thead>
        <tbody>{Object.entries(h.families).map(([f, d]: [string, any]) => (
          <tr key={f}><td>{d.title}</td>
            <td>{d.installed.length === 0 ? <span className="muted">not installed</span> :
              <select value={d.chosen && d.installed.includes(d.chosen) ? d.chosen : ""} onChange={(ev) => choose(f, ev.target.value)}>
                <option value="">Automatic ({d.auto} for this computer)</option>
                {d.installed.map((sz: string) => <option key={sz} value={sz}>{sz}</option>)}
              </select>}</td>
            <td>{d.train_auto ?? <span className="muted">–</span>}</td>
            <td className="muted">{d.installed.join(", ") || "–"}</td></tr>))}</tbody>
      </table>
      <p className="muted" style={{ marginBottom: 0 }}>YOLOX runs in missions with the standard detector, and RF-DETR in missions that choose
        it, unless a model is promoted. A promoted or saved model always runs as it is.</p>
      {err && <div className="msg err">{err}</div>}
    </div>
  );
}


const ACCEL: Record<string, string> = { cuda: "NVIDIA GPU (CUDA)", directml: "GPU (DirectML)", cpu: "CPU" };

function System({ sys, msg, backupNow }: { sys: any; msg: any; backupNow: () => void }) {
  const b = sys.backup;
  return (
    <div className="card">
      <h2>This workstation and backups</h2>
      <p style={{ margin: "4px 0" }}>Version {sys.version}. Models run on: {sys.runtime ? <>{ACCEL[sys.runtime.accelerator]}{sys.runtime.notes.length > 0 && <span className="muted"> ({sys.runtime.notes.join("; ")})</span>}</> : <span className="muted">decided when the first job loads the detector</span>}.</p>
      <p style={{ margin: "4px 0" }}>
        {b.folder ? <>Nightly database backup at {b.at} to <span className="mono">{b.folder}</span>. </> : <>No backup folder is set: start the server with <span className="mono">--backup-dir</span>. </>}
        {b.last ? <>Last: {b.last.at} ({b.last.detail.folder ?? b.last.detail.error}).</> : "No backup yet."}
        {b.last_error && <span className="msg err">Last scheduled backup failed: {b.last_error}</span>}
      </p>
      <div className="row"><button disabled={!b.folder} title={!b.folder ? "Set --backup-dir when starting the server to enable backups." : undefined} onClick={backupNow}>Back up now</button>{msg && <span className={`msg ${msg.kind}`} role={msg.kind === "err" ? "alert" : "status"} aria-live="polite">{msg.text}</span>}</div>
      {sys.audit && sys.audit.length > 0 && <details style={{ marginTop: 8 }}><summary>Audit log</summary>
        <ul className="muted">{sys.audit.map((e: any) => <li key={e.id}><span className="mono">{e.at}</span> {e.author}: {e.action.replace(/_/g, " ")}{e.detail.name ? ` “${e.detail.name}”` : ""}{e.detail.folder ? ` ${e.detail.folder}` : ""}</li>)}</ul></details>}
    </div>
  );
}
