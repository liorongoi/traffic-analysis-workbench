import { useEffect, useMemo, useState } from "react";
import { api, ApiError, post } from "../api";
import { go } from "../router";
import TimezoneMap from "../components/TimezoneMap";

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
    name: "", input_type: "images", mode: "general_traffic", country: "IL", iana_zone: "Asia/Jerusalem",
    latitude: null as number | null, longitude: null as number | null,
    source_dir: "", include_subdirs: true, processing: "one_time", output_dir: "", export_formats: ["xlsx", "csv"],
    detector: "standard",
  });
  const [zoneFilter, setZoneFilter] = useState("");
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
  const zones = useMemo(() => (ref?.zones ?? []).filter((z) => z.toLowerCase().includes(zoneFilter.toLowerCase())).slice(0, 200), [ref, zoneFilter]);

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
      setPreview((await post("/source-preview", { source_dir: cfg.source_dir, include_subdirs: cfg.include_subdirs })).files);
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
  const canNext = [cfg.name.trim() !== "", !!cfg.iana_zone, cfg.source_dir.trim() !== "", cfg.output_dir.trim() !== "" && cfg.export_formats.length > 0, true][step];

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
            <label className="field"><span>Mission name</span><input autoFocus value={cfg.name} onChange={(e) => set("name", e.target.value)} placeholder="e.g. Route 6 north, October" /></label>
            <label className="field"><span>Input type (cannot change later)</span>
              <select value={cfg.input_type} onChange={(e) => { set("input_type", e.target.value); set("mode", ref.modes[e.target.value][0]); if (e.target.value === "video") set("detector", "standard"); }}>
                <option value="images">Still images</option><option value="video">Video recordings</option>
              </select>
            </label>
            <label className="field"><span>Analysis mode (cannot change later)</span>
              <select value={cfg.mode} onChange={(e) => set("mode", e.target.value)}>
                {ref.modes[cfg.input_type].map((m) => <option key={m} value={m}>{m.replace(/_/g, " ")}</option>)}
              </select>
            </label>
            <p className="muted">{MODE_HELP[cfg.mode]}</p>
            {cfg.input_type === "images" && <DetectorChoice value={cfg.detector} onChange={(d) => set("detector", d)} sam={ref.sam3} />}
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
            <label className="field"><span>Search by city or IANA time-zone name</span>
              <input placeholder="e.g. Tehran, Beirut, Europe/Riga" value={zoneFilter} onChange={(e) => setZoneFilter(e.target.value)} />
              <select size={5} value={cfg.iana_zone} onChange={(e) => setCfg((c) => ({
                ...c, iana_zone: e.target.value, latitude: null, longitude: null,
              }))}>
                {!zones.includes(cfg.iana_zone) && <option value={cfg.iana_zone}>{cfg.iana_zone}</option>}
                {zones.map((z) => <option key={z} value={z}>{z} · {z.split("/").pop()?.replace(/_/g, " ")}</option>)}
              </select>
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
            <label className="field"><span>Source folder on this workstation (read only; files are never changed)</span>
              <input className="mono" value={cfg.source_dir} onChange={(e) => { set("source_dir", e.target.value); setPreview(null); }} placeholder="D:\Feeds\Route6" />
            </label>
            <label className="row"><input type="checkbox" checked={cfg.include_subdirs} onChange={(e) => set("include_subdirs", e.target.checked)} /> Include subfolders</label>
            <label className="field" style={{ marginTop: 10 }}><span>Processing</span>
              <select value={cfg.processing} onChange={(e) => set("processing", e.target.value)}>
                <option value="one_time">One-time: the files there now</option>
                <option value="continuous">Continuous: keep watching for new files</option>
              </select>
            </label>
            <div className="row"><button onClick={loadPreview} disabled={!cfg.source_dir}>Preview file names</button>
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
            <label className="field"><span>Output folder for exports</span>
              <input className="mono" value={cfg.output_dir} onChange={(e) => set("output_dir", e.target.value)} placeholder="E:\Exports\Route6" />
            </label>
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
                Source: `${cfg.source_dir}${cfg.include_subdirs ? " (with subfolders)" : ""}`, Processing: cfg.processing.replace("_", "-"), Output: cfg.output_dir, Formats: cfg.export_formats.join(", "),
                "Vehicle detector": cfg.detector === "sam3" ? "SAM 3.1 (finds vehicles by name)" : cfg.detector === "both" ? "Both: standard and SAM 3.1, merged" : "Standard (YOLOX-s)" })
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

export function DetectorChoice({ value, onChange, sam, disabled, legend = true }: {
  value: string; onChange: (d: string) => void; sam?: { installed: boolean; reason: string | null; memory_note?: string | null };
  disabled?: boolean; legend?: boolean;
}) {
  return (
    <fieldset style={{ border: 0, padding: 0, margin: "4px 0 14px", minWidth: 0 }} aria-label="Vehicle detector">
      {legend && <legend style={{ padding: 0, fontSize: 12.5, fontWeight: 550, color: "var(--text-2)", marginBottom: 6 }}>Vehicle detector (can change later)</legend>}
      <div className="choices">
        <label className="choice">
          <input type="radio" name="detector" checked={value === "standard"} disabled={disabled} onChange={() => onChange("standard")} />
          <b>Standard</b>
          <small>YOLOX-s, or the model promoted on the learning page. About a second per image, and it can be trained on your own camera images.</small>
        </label>
        <label className="choice">
          <input type="radio" name="detector" checked={value === "sam3"} disabled={disabled || !sam?.installed} onChange={() => onChange("sam3")} />
          <b>SAM 3.1</b>
          <small>{sam?.installed
            ? "Meta's Segment Anything 3.1 asks for each kind of vehicle by name. More thorough on unusual angles and crowded scenes; on a CPU it takes from several seconds to a minute per image."
            : `Not available on this computer. ${sam?.reason ?? ""}`}</small>
        </label>
        <label className="choice">
          <input type="radio" name="detector" checked={value === "both"} disabled={disabled || !sam?.installed} onChange={() => onChange("both")} />
          <b>Both</b>
          <small>{sam?.installed
            ? "The standard detector and SAM 3.1 on every image. A vehicle both find counts once; each vehicle says which of them found it. The most complete, and as slow as SAM 3.1."
            : "Needs SAM 3.1, which is not available on this computer."}</small>
        </label>
      </div>
      {(value === "sam3" || value === "both") && sam?.memory_note && <div className="msg warn">{sam.memory_note}</div>}
    </fieldset>
  );
}
