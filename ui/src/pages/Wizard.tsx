import { useEffect, useMemo, useState } from "react";
import { api, ApiError, post } from "../api";
import { go } from "../router";
import TimezoneMap from "../components/TimezoneMap";
import FolderPicker from "../components/FolderPicker";
import { NamePick, ZonePick, thisMonth } from "../components/Pick";

type Ref = { countries: Record<string, string>; country_zones: Record<string, string>; modes: Record<string, string[]>; zones: string[];
  sam3?: { installed: boolean; reason: string | null; memory_note?: string | null } };
type Preview = { file: string; ok: boolean; utc?: string; precision?: string; pattern?: string; reason?: string };

const STEPS = ["Mission", "Place and time", "Source", "Output", "Check and create"];
const MODE_HELP: Record<string, string> = {
  general_traffic: "Vehicles per image, within the view's regions. Never presented as unique vehicles or traffic flow.",
  object_search: "Find vehicles matching a description.",
  plate_recognition: "Read licence plates, country by country. In video, each vehicle's plate is read on several frames.",
};

export default function Wizard() {
  const [ref, setRef] = useState<Ref | null>(null);
  const [refError, setRefError] = useState("");
  const [step, setStep] = useState(0);
  const [cfg, setCfg] = useState({
    name: `Traffic, ${thisMonth()}`, input_type: "images", mode: "general_traffic", country: "IL", iana_zone: "Asia/Jerusalem",
    latitude: null as number | null, longitude: null as number | null,
    source_dir: "", source_dirs: [] as string[], include_subdirs: true, processing: "one_time", output_dir: "", export_formats: ["xlsx", "csv"],
    detector: "standard", vehicle_model: null as string | null, sam3_interval_s: null as number | null, camera: "fixed", video_workers: null as number | null,
  });
  const [saved, setSaved] = useState<any[]>([]);
  useEffect(() => { api("/models/saved").then((r) => setSaved(r.saved ?? [])).catch(() => setSaved([])); }, []);
  const [utcPreview, setUtcPreview] = useState(() => new Date().toISOString().slice(0, 16));
  const [timePreview, setTimePreview] = useState<{ local: string; offset: string; is_dst: boolean; abbreviation: string } | null>(null);
  const [preview, setPreview] = useState<Preview[] | null>(null);
  const [check, setCheck] = useState<{ errors: string[]; warnings: string[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    api("/reference").then((result) => { if (live) { setRef(result); setRefError(""); } }).catch((e) => {
      if (live) setRefError(e instanceof Error ? e.message : "Could not load setup options.");
    });
    return () => { live = false; };
  }, []);
  const set = (k: string, v: unknown) => setCfg((c) => ({ ...c, [k]: v }));
  // Names to choose from: what the mission looks at, where and when. The first is used unless another is chosen.
  const nameIdeas = useMemo(() => {
    const what = { general_traffic: "Traffic", object_search: "Vehicle search", plate_recognition: "Plates" }[cfg.mode] ?? "Mission";
    const place = ref?.countries[cfg.country] ?? "";
    const city = cfg.iana_zone.split("/").pop()?.replace(/_/g, " ") ?? "";
    const folders = cfg.source_dirs.map((d) => d.split(/[\\/]/).filter(Boolean).pop() ?? "");
    const kind = cfg.input_type === "video" ? "video" : "";
    return [`${what}, ${thisMonth()}`, ...folders.map((f) => `${f}, ${thisMonth()}`), `${what}, ${city}`, place && `${what}, ${place}`,
      kind && `${what} ${kind}, ${thisMonth()}`, `${city} ${what.toLowerCase()}, ${thisMonth()}`].filter(Boolean) as string[];
  }, [ref, cfg.mode, cfg.country, cfg.iana_zone, cfg.source_dirs, cfg.input_type]);

  useEffect(() => {
    if (step === 4) {
      setCheck(null);
      setErr(null);
      post("/missions/validate", cfg).then(setCheck).catch((e: ApiError) => setErr(e.message));
    }
  }, [step]);

  useEffect(() => {
    let live = true;
    const utc = new Date(`${utcPreview}:00Z`);
    if (Number.isNaN(utc.valueOf())) { setTimePreview(null); return; }
    post("/time-preview", { iana_zone: cfg.iana_zone, utc: utc.toISOString() })
      .then((result) => { if (live) setTimePreview(result); })
      .catch(() => { if (live) setTimePreview(null); });
    return () => { live = false; };
  }, [cfg.iana_zone, utcPreview]);

  const loadPreview = async () => {
    setErr(null);
    try {
      // A few names from each chosen folder.
      const per = Math.max(5, Math.floor(20 / Math.max(cfg.source_dirs.length, 1)));
      const all: Preview[] = [];
      for (const d of cfg.source_dirs) {
        const files: Preview[] = (await post("/source-preview", { source_dir: d, include_subdirs: cfg.include_subdirs, limit: per })).files;
        all.push(...files.map((f) => cfg.source_dirs.length > 1 ? { ...f, file: `${d.split(/[\\/]/).filter(Boolean).pop()}/${f.file}` } : f));
      }
      setPreview(all);
    } catch (e) {
      setPreview(null);
      setErr((e as ApiError).message);
    }
  };

  const create = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await post("/missions", cfg);
      go(`/m/${r.id}`);
    } catch (e) {
      setErr((e as ApiError).message);
    } finally {
      setBusy(false);
    }
  };

  if (!ref) return <div className="page">
    {refError ? <div className="card msg err" role="alert"><b>Setup options could not be loaded.</b><p>{refError}</p><button onClick={() => { setRefError(""); api("/reference").then(setRef).catch((e) => setRefError(e instanceof Error ? e.message : "Could not load setup options.")); }}>Try again</button></div>
      : <p className="muted" role="status" aria-live="polite">Loading setup options…</p>}
  </div>;
  const canNext = [cfg.name.trim() !== "", !!cfg.iana_zone, cfg.source_dirs.length > 0, cfg.output_dir.trim() !== "" && cfg.export_formats.length > 0, true][step];

  return (
    <div className="page" style={{ maxWidth: 820 }}>
      <h1>New mission</h1>
      <ol className="steps" aria-label="Steps">
        {STEPS.map((s, i) => (
          <li key={s} className={i === step ? "on" : i < step ? "done" : ""} aria-current={i === step ? "step" : undefined}><b>{i + 1}</b><span>{s}</span></li>
        ))}
      </ol>
      <div className="card">
        {step === 0 && (
          <>
            <label className="field"><span>Mission name</span><NamePick label="Mission name" value={cfg.name} onChange={(v) => set("name", v)} suggestions={nameIdeas} /></label>
            <label className="field"><span>Input type (cannot change later)</span>
              <select value={cfg.input_type} onChange={(e) => { set("input_type", e.target.value); set("mode", ref.modes[e.target.value][0]); }}>
                <option value="images">Still images</option><option value="video">Video recordings</option>
              </select>
            </label>
            <label className="field"><span>Analysis mode (cannot change later)</span>
              <select value={cfg.mode} onChange={(e) => set("mode", e.target.value)}>
                {ref.modes[cfg.input_type].map((m) => <option key={m} value={m}>{m.replace(/_/g, " ")}</option>)}
              </select>
            </label>
            <p className="muted">{MODE_HELP[cfg.mode]}</p>
            <DetectorChoice value={cfg.detector} onChange={(d) => set("detector", d)} sam={ref.sam3} video={cfg.input_type === "video"} />
            {cfg.input_type === "video" && <CameraChoice value={cfg.camera} onChange={(v) => set("camera", v)} />}
            {cfg.input_type === "video" && <ParallelChoice value={cfg.video_workers} onChange={(v) => set("video_workers", v)} />}
            {cfg.input_type === "video" && usesSam(cfg.detector) && <SamInterval value={cfg.sam3_interval_s} onChange={(v) => set("sam3_interval_s", v)} both={cfg.detector === "both"} />}
            {cfg.detector !== "sam3" && <ModelChoice saved={saved} value={cfg.vehicle_model} onChange={(v) => set("vehicle_model", v)} rfdetr={cfg.detector === "rfdetr"} />}
          </>
        )}
        {step === 1 && (
          <>
            <label className="field"><span>Country</span>
              <select value={cfg.country} onChange={(e) => {
                const country = e.target.value;
                setCfg((c) => ({ ...c, country, iana_zone: ref.country_zones[country] ?? c.iana_zone, latitude: null, longitude: null }));
              }}>
                {Object.entries(ref.countries).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                <option value="FI">Finland (sample data; no plate reading)</option>
              </select>
            </label>
            <label className="field"><span>Time zone: area, then city (or click the map)</span>
              <ZonePick zones={ref.zones} value={cfg.iana_zone} onChange={(z) => setCfg((c) => ({ ...c, iana_zone: z, latitude: null, longitude: null }))} />
            </label>
            <TimezoneMap selected={cfg.iana_zone} onPick={(zone, latitude, longitude) => {
              setCfg((c) => ({ ...c, iana_zone: zone, latitude, longitude }));
            }} />
            <div className="row muted">Selected zone: <b>{cfg.iana_zone}</b>
              {cfg.latitude !== null && cfg.longitude !== null && <span> · Location: {cfg.latitude.toFixed(4)}, {cfg.longitude.toFixed(4)}</span>}
            </div>
            <label className="field" style={{ marginTop: 10 }}><span>Check local time for a UTC capture date</span>
              <input type="datetime-local" value={utcPreview} onChange={(e) => setUtcPreview(e.target.value)} />
            </label>
            {timePreview && <p className="msg ok">{timePreview.local.replace("T", " ")} {timePreview.abbreviation} ({timePreview.offset}); daylight saving {timePreview.is_dst ? "applies" : "does not apply"} on this date.</p>}
          </>
        )}
        {step === 2 && (
          <>
            <FolderPicker multiple label="Source folders on this workstation (read only; files are never changed). Add one or several."
              value={cfg.source_dirs} placeholder="Click to choose a folder"
              onChange={(v) => { setCfg((c) => ({ ...c, source_dirs: v, source_dir: v[0] ?? "" })); setPreview(null); }} />
            <label className="row"><input type="checkbox" checked={cfg.include_subdirs} onChange={(e) => set("include_subdirs", e.target.checked)} /> Include subfolders</label>
            <label className="field" style={{ marginTop: 10 }}><span>Processing</span>
              <select value={cfg.processing} onChange={(e) => set("processing", e.target.value)}>
                <option value="one_time">One-time: the files there now</option>
                <option value="continuous">Continuous: keep watching for new files</option>
              </select>
            </label>
            <div className="row"><button onClick={loadPreview} disabled={cfg.source_dirs.length === 0}>Preview file names</button>
              <span className="muted">Capture time comes from the file name in UTC. File dates are never used.</span></div>
            {preview && (
              <div className="tablewrap" style={{ marginTop: 10, maxHeight: 260, overflowY: "auto" }}>
                <table className="t">
                  <thead><tr><th>File</th><th>Capture time (UTC)</th><th>Precision</th><th>Result</th></tr></thead>
                  <tbody>{preview.map((p) => (
                    <tr key={p.file}><td className="mono bidi">{p.file}</td><td className="mono">{p.utc ?? ""}</td><td>{p.precision ?? ""}</td>
                      <td>{p.ok ? <span className="badge approved">read</span> : <span className="badge failed" title={p.reason}>not readable: {p.reason}</span>}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            )}
            {preview?.some((p) => !p.ok) && (
              <div className="msg warn">Some names have no capture time the app can read, so those files would be skipped.{" "}
                <a href="#/formats" target="_blank" rel="noreferrer">Show the app how your files are named</a> (opens in a new tab), then press Preview file names again.</div>
            )}
          </>
        )}
        {step === 3 && (
          <>
            <FolderPicker label="Output folder for exports" value={cfg.output_dir ? [cfg.output_dir] : []} placeholder="Click to choose a folder"
              onChange={(v) => set("output_dir", v[0] ?? "")} />
            <div className="row">
              {["xlsx", "csv"].map((f) => (
                <label key={f} className="row"><input type="checkbox" checked={cfg.export_formats.includes(f)}
                  onChange={(e) => set("export_formats", e.target.checked ? [...cfg.export_formats, f] : cfg.export_formats.filter((x) => x !== f))} />{f === "xlsx" ? "Excel (.xlsx)" : "CSV"}</label>
              ))}
            </div>
          </>
        )}
        {step === 4 && (
          <>
            <table className="t"><tbody>
              {Object.entries({ Name: cfg.name, "Input type": cfg.input_type, Mode: cfg.mode.replace(/_/g, " "), Country: cfg.country, "Time zone": cfg.iana_zone,
                Location: cfg.latitude === null || cfg.longitude === null ? "Not set" : `${cfg.latitude.toFixed(4)}, ${cfg.longitude.toFixed(4)}`,
                [cfg.source_dirs.length > 1 ? "Sources" : "Source"]: `${cfg.source_dirs.join("; ")}${cfg.include_subdirs ? " (with subfolders)" : ""}`, Processing: cfg.processing.replace("_", "-"), Output: cfg.output_dir, Formats: cfg.export_formats.join(", "),
                Detector: (DETECTOR_LABEL[cfg.detector] ?? cfg.detector)
                  + (cfg.input_type === "video" && usesSam(cfg.detector) ? `, SAM 3.1 every ${cfg.sam3_interval_s ?? 1} s of video` : ""),
                ...(cfg.input_type === "video" ? { Camera: cfg.camera === "moving" ? "Moving: camera motion is measured and removed" : "Fixed" } : {}),
                ...(cfg.detector !== "sam3" ? { Model: cfg.vehicle_model ? `${cfg.vehicle_model} (saved model)` : "The model in production use" } : {}) })
                .map(([k, v]) => <tr key={k}><th style={{ width: 140 }}>{k}</th><td className="bidi">{v}</td></tr>)}
            </tbody></table>
            {!check && !err && <p className="muted" role="status" aria-live="polite">Checking folders and settings…</p>}
            {check?.errors.map((e) => <div key={e} className="msg err" role="alert">{e}</div>)}
            {check?.warnings.map((w) => <div key={w} className="msg warn">{w}</div>)}
            {check && check.errors.length === 0 && <div className="msg ok">Checks passed. The mission database is created when you press Create.</div>}
          </>
        )}
        {err && <div className="msg err" role="alert">{err}</div>}
        <div className="row" style={{ marginTop: 14, justifyContent: "space-between" }}>
          <button onClick={() => (step === 0 ? go("/") : setStep(step - 1))}>{step === 0 ? "Cancel" : "Back"}</button>
          {step < 4 ? <button className="primary" disabled={!canNext} onClick={() => setStep(step + 1)}>Next</button>
            : <button className="primary" disabled={busy || !check || check.errors.length > 0} onClick={create}>{busy ? "Creating…" : "Create mission"}</button>}
        </div>
      </div>
    </div>
  );
}

export const DETECTOR_LABEL: Record<string, string> = {
  standard: "Standard (YOLOX-s)", rfdetr: "RF-DETR (transformer detector)", sam3: "SAM 3.1 (finds objects by name)",
  both: "Both: standard and SAM 3.1, merged",
};
/** SAM 3.1 runs for these choices (so its interval and words apply). */
export const usesSam = (d: string) => d === "sam3" || d === "both";

export type Grades = {
  measured?: string; machine?: string; note?: string;
  scenes: { key: string; label: string; short?: string; set: string }[];
  detectors: Record<string, { model: string; ap50: Record<string, number | null>; seconds: Record<string, number | null>;
    trained?: Record<string, number | null>; trained_note?: string }>;
};
type DetectorInfo = { rfdetr?: { installed: boolean; reason: string | null; model?: { name: string; version: string } | null }; grades?: Grades };

/** Letter grades: vehicle AP50 (share of vehicles found with a well-placed box, ranked by confidence) and speed. */
export function apGrade(v: number | null | undefined): string {
  if (v == null) return "–";
  return v >= 80 ? "A" : v >= 65 ? "B" : v >= 50 ? "C" : v >= 30 ? "D" : "F";
}
export function speedGrade(s: number | null | undefined): string {
  if (s == null) return "–";
  return s <= 0.15 ? "A" : s <= 0.4 ? "B" : s <= 1.5 ? "C" : s <= 5 ? "D" : "F";
}
const meanSeconds = (d: Record<string, number | null>) => {
  const v = Object.values(d).filter((x): x is number => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
const fmtS = (s: number | null) => (s == null ? "–" : s < 1 ? `${s.toFixed(2)} s` : `${s.toFixed(1)} s`);

/** One line of grades under a detector's name: per kind of scene, then speed. */
export function GradeLine({ g, k }: { g?: Grades; k: string }) {
  const d = g?.detectors?.[k];
  if (!g || !d) return null;
  const s = meanSeconds(d.seconds);
  return (
    <span className="grades" aria-label="Benchmark grades">
      {g.scenes.map((sc) => (
        <span key={sc.key} className={`grade g-${apGrade(d.ap50[sc.key])}`} title={`${sc.label}: vehicle AP50 ${d.ap50[sc.key] ?? "not measured"} (${sc.set})`}>
          {sc.short ?? sc.label} <b>{apGrade(d.ap50[sc.key])}</b>
        </span>
      ))}
      <span className={`grade g-${speedGrade(s)}`} title={`Seconds per image on ${g.machine ?? "a CPU"}, averaged over the scenes`}>Speed <b>{speedGrade(s)}</b></span>
    </span>
  );
}

/** The measured scores behind the grades, all detectors side by side, the best in each column marked. */
export function GradesTable({ g }: { g: Grades }) {
  const keys = ["standard", "rfdetr", "sam3", "both"].filter((k) => g.detectors[k]);
  const best = (sc: string) => Math.max(...keys.map((k) => g.detectors[k].ap50[sc] ?? -1));
  const fastest = Math.min(...keys.map((k) => meanSeconds(g.detectors[k].seconds) ?? 1e9));
  const trained = keys.filter((k) => g.detectors[k].trained);
  return (
    <details className="grades-table">
      <summary>How these grades were measured</summary>
      <div style={{ overflowX: "auto" }}>
        <table className="t">
          <thead><tr><th>Detector</th>{g.scenes.map((sc) => <th key={sc.key} title={sc.set}>{sc.label}</th>)}<th>Seconds per image</th></tr></thead>
          <tbody>{keys.map((k) => {
            const d = g.detectors[k];
            const s = meanSeconds(d.seconds);
            return (
              <tr key={k}><th>{DETECTOR_LABEL[k]?.split(" (")[0] ?? k}<br /><small className="muted">{d.model}</small></th>
                {g.scenes.map((sc) => <td key={sc.key} style={{ fontWeight: d.ap50[sc.key] === best(sc.key) ? 650 : undefined }}>
                  {d.ap50[sc.key] == null ? "–" : d.ap50[sc.key]!.toFixed(1)} <span className={`grade g-${apGrade(d.ap50[sc.key])}`}><b>{apGrade(d.ap50[sc.key])}</b></span></td>)}
                <td style={{ fontWeight: s === fastest ? 650 : undefined }}>{fmtS(s)} <span className={`grade g-${speedGrade(s)}`}><b>{speedGrade(s)}</b></span></td>
              </tr>
            );
          })}</tbody>
        </table>
      </div>
      {trained.length > 0 && <>
        <p className="muted" style={{ margin: "10px 0 4px" }}>After training on 300 thermal drone images (vehicle AP50, same thermal test images):</p>
        <table className="t"><tbody>{trained.map((k) => (
          <tr key={k}><th>{DETECTOR_LABEL[k]?.split(" (")[0] ?? k}</th>
            {Object.entries(g.detectors[k].trained!).map(([sc, v]) => <td key={sc}>{v == null ? "–" : v.toFixed(1)} <span className={`grade g-${apGrade(v)}`}><b>{apGrade(v)}</b></span></td>)}
            <td className="muted">{g.detectors[k].trained_note ?? ""}</td></tr>))}</tbody></table>
      </>}
      <p className="muted" style={{ fontSize: 12.5 }}>
        Vehicle AP50 in percent: how many of the vehicles each detector finds with a well-placed box, ranked by its confidence
        (A 80 or more, B 65, C 50, D 30, F below). {g.scenes.map((sc) => `${sc.label}: ${sc.set}.`).join(" ")} Speed on {g.machine ?? "a CPU"}{" "}
        (A up to 0.15 s per image, B 0.4 s, C 1.5 s, D 5 s, F slower); a graphics card is many times faster. Your own cameras
        can differ: compare detectors on your reviewed images on the learning page.{g.measured ? ` Measured ${g.measured}.` : ""}
      </p>
    </details>
  );
}

export function DetectorChoice({ value, onChange, sam, disabled, legend = true, video = false }: {
  value: string; onChange: (d: string) => void; sam?: { installed: boolean; reason: string | null; memory_note?: string | null };
  disabled?: boolean; legend?: boolean; video?: boolean;
}) {
  const unit = video ? "analysed frame" : "image";
  const [info, setInfo] = useState<DetectorInfo>({});
  useEffect(() => { api("/detectors").then(setInfo).catch(() => setInfo({})); }, []);
  const rf = info.rfdetr;
  const g = info.grades && info.grades.scenes ? info.grades : undefined;
  return (
    <fieldset style={{ border: 0, padding: 0, margin: "4px 0 14px", minWidth: 0 }} aria-label="Detector">
      {legend && <legend style={{ padding: 0, fontSize: 12.5, fontWeight: 550, color: "var(--text-2)", marginBottom: 6 }}>Detector (can change later)</legend>}
      <div className="choices">
        <label className="choice">
          <input type="radio" name="detector" checked={value === "standard"} disabled={disabled} onChange={() => onChange("standard")} />
          <b>Standard</b>
          <GradeLine g={g} k="standard" />
          <small>YOLOX-s, or the model promoted on the learning page. The fastest, and it can be trained on your own camera {video ? "recordings" : "images"}. Finds vehicles, and people when the mission looks for them; weak on small targets from the air and in thermal.</small>
        </label>
        <label className="choice">
          <input type="radio" name="detector" checked={value === "rfdetr"} disabled={disabled || (rf != null && !rf.installed)} onChange={() => onChange("rfdetr")} />
          <b>RF-DETR</b>
          <GradeLine g={g} k="rfdetr" />
          <small>{rf == null || rf.installed
            ? `A newer transformer detector (Roboflow, Apache-2.0)${rf?.model ? `, ${rf.model.name}` : ""}. Finds more vehicles than YOLOX-s on every kind of scene, at about two to three times its time per ${unit}, and it can be trained on your own ${video ? "recordings" : "images"} the same way.`
            : `Not available on this computer. ${rf.reason ?? ""}`}</small>
        </label>
        <label className="choice">
          <input type="radio" name="detector" checked={value === "sam3"} disabled={disabled || !sam?.installed} onChange={() => onChange("sam3")} />
          <b>SAM 3.1</b>
          <GradeLine g={g} k="sam3" />
          <small>{sam?.installed
            ? (video ? "Meta's Segment Anything 3.1 asks for each kind of object by name, every second of video by default; the tracker follows what it found in between. Finds anything you can name, and is slow on a CPU."
              : "Meta's Segment Anything 3.1 asks for each kind of object by name. More thorough on unusual angles and crowded scenes; on a CPU it takes from several seconds to a minute per image.")
            : `Not available on this computer. ${sam?.reason ?? ""}`}</small>
        </label>
        <label className="choice">
          <input type="radio" name="detector" checked={value === "both"} disabled={disabled || !sam?.installed} onChange={() => onChange("both")} />
          <b>Both</b>
          <GradeLine g={g} k="both" />
          <small>{sam?.installed
            ? (video ? "The standard detector on every analysed frame and SAM 3.1 every second of video by default, merged. An object both find counts once; each tracked box says which of them found it."
              : "The standard detector and SAM 3.1 on every image. A vehicle both find counts once; each vehicle says which of them found it. The most complete, and as slow as SAM 3.1.")
            : "Needs SAM 3.1, which is not available on this computer."}</small>
        </label>
      </div>
      {g && <GradesTable g={g} />}
      {usesSam(value) && sam?.memory_note && <div className="msg warn">{sam.memory_note}</div>}
    </fieldset>
  );
}

/** Video: does the camera move? A moving camera (PTZ, on a vehicle, a drone, worn) has its own motion measured on
 *  every analysed frame and taken out before tracking, so a pan neither breaks tracks nor counts as a cut. */
export function CameraChoice({ value, onChange, disabled }: { value: string; onChange: (v: string) => void; disabled?: boolean }) {
  return (
    <label className="field"><span>Camera</span>
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        <option value="fixed">Fixed: mounted and not moving (fastest)</option>
        <option value="moving">Moving: PTZ, on a vehicle, a drone or worn</option>
      </select>
      <small className="muted">{value === "moving" ? "The camera's own motion is measured on every analysed frame (a few milliseconds) and removed, so targets keep one track while the camera pans or zooms." : "A sudden change of the whole picture is treated as a scene cut."}</small>
    </label>
  );
}

/** Video: how many pieces of a long recording are analysed side by side on this computer. */
export function ParallelChoice({ value, onChange, disabled }: { value: number | null; onChange: (v: number) => void; disabled?: boolean }) {
  const n = value ?? 1;
  return (
    <label className="field"><span>Analyse side by side</span>
      <select value={String(n)} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))}>
        {[1, 2, 3, 4, 6, 8].map((k) => <option key={k} value={k}>{k === 1 ? "1 piece at a time" : `${k} pieces at a time`}</option>)}
      </select>
      <small className="muted">{n === 1 ? "A recording is analysed from start to end." : "Recordings longer than about 6 minutes are cut into 5-minute pieces with a few seconds of overlap, analysed at the same time and joined at the cuts, so every target is counted once. Use more pieces on a computer with a strong graphics card or many cores."}</small>
    </label>
  );
}

/** Video: how often SAM 3.1 looks at the recording. Between those frames the tracker (and, with "Both", the
 *  standard detector) follows what it found. */
export function SamInterval({ value, onChange, both, disabled }: { value: number | null; onChange: (v: number) => void; both: boolean; disabled?: boolean }) {
  return (
    <label className="field"><span>SAM 3.1 looks at the video every</span>
      <select value={String(value ?? 1)} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))}>
        {[0.2, 0.5, 1, 2, 5].map((s) => <option key={s} value={s}>{s === 0.2 ? "0.2 s (every analysed frame, slowest)" : `${s} s`}</option>)}
      </select>
      <small className="muted">{both ? "The standard detector still looks at every analysed frame (5 per second)." : "Only these frames are analysed; a shorter interval follows fast objects better."}</small>
    </label>
  );
}

/** Which model the standard detector (or RF-DETR) uses: the default one, or a model saved by name on a learning page. */
export function ModelChoice({ saved, value, onChange, disabled, rfdetr = false }: {
  saved: any[]; value: string | null; onChange: (v: string | null) => void; disabled?: boolean; rfdetr?: boolean;
}) {
  // Saved models of the chosen kind, plus the one already chosen whatever its kind.
  const usable = saved.filter((x) => x.installed && (Boolean(x.rfdetr) === rfdetr || x.name === value));
  return (
    <label className="field"><span>{rfdetr ? "Model for RF-DETR" : "Model for the standard detector"}</span>
      <select value={value ?? ""} disabled={disabled} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">{rfdetr ? "The released RF-DETR (or the RF-DETR promoted on the learning page)" : "The model in production use"}</option>
        {usable.map((x) => <option key={x.id} value={x.name}>{x.name}{x.description ? ` – ${x.description}` : ""}</option>)}
      </select>
      <small className="muted" style={{ fontSize: 12.5 }}>{usable.length ? "Saved models are frozen models kept by name on a mission's learning page, for example one trained on cars at night."
        : "No saved models yet. Train one on a mission's learning page and save it by name to choose it here."}</small>
    </label>
  );
}
