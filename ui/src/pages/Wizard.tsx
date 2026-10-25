import { useEffect, useMemo, useState } from "react";
import { api, ApiError, post } from "../api";
import { go } from "../router";

type Ref = { countries: Record<string, string>; country_zones: Record<string, string>; modes: Record<string, string[]>; zones: string[] };
type Preview = { file: string; ok: boolean; utc?: string; precision?: string; pattern?: string; reason?: string };

const STEPS = ["Mission", "Place and time", "Source", "Output", "Check and create"];
const MODE_HELP: Record<string, string> = {
  general_traffic: "Vehicles per image, within the view's regions. Never presented as unique vehicles or traffic flow.",
  object_search: "Find vehicles matching a description.",
  plate_recognition: "Read licence plates, country by country. In video, each vehicle's plate is read on several frames.",
};

export default function Wizard() {
  const [ref, setRef] = useState<Ref | null>(null);
  const [step, setStep] = useState(0);
  const [cfg, setCfg] = useState({
    name: "", input_type: "images", mode: "general_traffic", country: "IL", iana_zone: "Asia/Jerusalem",
    source_dir: "", include_subdirs: true, processing: "one_time", output_dir: "", export_formats: ["xlsx", "csv"],
  });
  const [zoneFilter, setZoneFilter] = useState("");
  const [preview, setPreview] = useState<Preview[] | null>(null);
  const [check, setCheck] = useState<{ errors: string[]; warnings: string[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { api("/reference").then(setRef); }, []);
  const set = (k: string, v: unknown) => setCfg((c) => ({ ...c, [k]: v }));
  const zones = useMemo(() => (ref?.zones ?? []).filter((z) => z.toLowerCase().includes(zoneFilter.toLowerCase())).slice(0, 200), [ref, zoneFilter]);

  useEffect(() => {
    if (step === 4) post("/missions/validate", cfg).then(setCheck).catch((e: ApiError) => setErr(e.message));
  }, [step]);

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

  if (!ref) return <div className="page muted">Loading…</div>;
  const canNext = [cfg.name.trim() !== "", !!cfg.iana_zone, cfg.source_dir.trim() !== "", cfg.output_dir.trim() !== "" && cfg.export_formats.length > 0, true][step];

  return (
    <div className="page" style={{ maxWidth: 820 }}>
      <h1>New mission</h1>
      <div className="row" style={{ marginBottom: 12 }}>
        {STEPS.map((s, i) => (
          <span key={s} className="badge" style={i === step ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}>{i + 1}. {s}</span>
        ))}
      </div>
      <div className="card">
        {step === 0 && (
          <>
            <label className="field"><span>Mission name</span><input autoFocus value={cfg.name} onChange={(e) => set("name", e.target.value)} placeholder="e.g. Route 6 north, October" /></label>
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
          </>
        )}
        {step === 1 && (
          <>
            <label className="field"><span>Country</span>
              <select value={cfg.country} onChange={(e) => { set("country", e.target.value); if (ref.country_zones[e.target.value]) set("iana_zone", ref.country_zones[e.target.value]); }}>
                {Object.entries(ref.countries).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                <option value="FI">Finland (sample data; no plate reading)</option>
              </select>
            </label>
            <label className="field"><span>Time zone (IANA). Local time and DST are worked out for each capture time.</span>
              <input placeholder="Filter zones…" value={zoneFilter} onChange={(e) => setZoneFilter(e.target.value)} />
              <select size={6} value={cfg.iana_zone} onChange={(e) => set("iana_zone", e.target.value)}>
                {!zones.includes(cfg.iana_zone) && <option value={cfg.iana_zone}>{cfg.iana_zone}</option>}
                {zones.map((z) => <option key={z} value={z}>{z}</option>)}
              </select>
            </label>
            <p className="muted">Selected: <b>{cfg.iana_zone}</b></p>
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
                Source: `${cfg.source_dir}${cfg.include_subdirs ? " (with subfolders)" : ""}`, Processing: cfg.processing.replace("_", "-"), Output: cfg.output_dir, Formats: cfg.export_formats.join(", ") })
                .map(([k, v]) => <tr key={k}><th style={{ width: 140 }}>{k}</th><td className="bidi">{v}</td></tr>)}
            </tbody></table>
            {check?.errors.map((e) => <div key={e} className="msg err">{e}</div>)}
            {check?.warnings.map((w) => <div key={w} className="msg warn">{w}</div>)}
            {check && check.errors.length === 0 && <div className="msg ok">Checks passed. The mission database is created when you press Create.</div>}
          </>
        )}
        {err && <div className="msg err">{err}</div>}
        <div className="row" style={{ marginTop: 14, justifyContent: "space-between" }}>
          <button onClick={() => (step === 0 ? go("/") : setStep(step - 1))}>{step === 0 ? "Cancel" : "Back"}</button>
          {step < 4 ? <button className="primary" disabled={!canNext} onClick={() => setStep(step + 1)}>Next</button>
            : <button className="primary" disabled={busy || !check || check.errors.length > 0} onClick={create}>Create mission</button>}
        </div>
      </div>
    </div>
  );
}
