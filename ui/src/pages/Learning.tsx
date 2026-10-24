import { Fragment, useCallback, useEffect, useState } from "react";
import { api, ApiError, post } from "../api";

// Learning, part 1: which reviewed images can teach a model, frozen dataset versions, evaluation of a model on them,
// and the operator's promotion of a model to production use. Nothing trains or promotes on its own.

const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
const num = (v: number | null | undefined, d = 2) => (v == null ? "–" : v.toFixed(d));

export default function LearningTab({ missionId, job, reload }: { missionId: string; job: any; reload: () => void }) {
  const [s, setS] = useState<any>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [name, setName] = useState("");
  const [ratios, setRatios] = useState("70/15/15");
  const [blockDays, setBlockDays] = useState(1);
  const [evalDs, setEvalDs] = useState<number | "">("");
  const [evalSplit, setEvalSplit] = useState("test");
  const [evalModel, setEvalModel] = useState("");
  const [open, setOpen] = useState<number | null>(null);

  const load = useCallback(async () => setS(await api(`/missions/${missionId}/learning`)), [missionId]);
  useEffect(() => { load(); }, [load]);
  const evaluating = job?.kind === "evaluate" && ["running", "paused"].includes(job.state);
  useEffect(() => {
    if (job?.kind !== "evaluate" || evaluating) return;
    load();
    setMsg((m) => (m?.text === "Evaluation started." ? (job.state === "finished" ? { kind: "ok", text: "Evaluation finished." }
      : { kind: "err", text: `Evaluation ${job.state}${job.error ? `: ${job.error}` : ""}.` }) : m));
  }, [job?.state, job?.kind, evaluating, load]);

  const act = async (fn: () => Promise<any>, ok?: (r: any) => string) => {
    setMsg(null);
    try { const r = await fn(); if (ok) setMsg({ kind: "ok", text: ok(r) }); await load(); reload(); }
    catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };

  if (!s) return <div className="muted">Loading…</div>;
  const e = s.eligibility;
  const prod = s.production;
  return (
    <div className="grid" style={{ gap: 10 }}>
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Images that can teach a model</h2>
        <div className="grid cols-4">
          <div className="card stat"><div className="n">{e.images.full}</div><div className="l">Approved one by one: full labels</div></div>
          <div className="card stat"><div className="n">{e.images.partial}</div><div className="l">Corrected, not approved: corrected objects only</div></div>
          <div className="card stat"><div className="n">{e.images.bulk}</div><div className="l">Bulk approved: never used</div></div>
          <div className="card stat"><div className="n">{e.images.none}</div><div className="l">Not reviewed: never used</div></div>
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          Approving an image on the investigation screen says every vehicle inside the included regions is marked, so the
          rest of the road there counts as “no vehicle”. Where you only corrected some objects, only those are used and the
          rest of the image is ignored. Unresolved night candidates are never counted as background.
        </p>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Dataset versions</h2>
        <div className="row">
          <input placeholder="Version name, e.g. October day" value={name} onChange={(ev) => setName(ev.target.value)} />
          <label>Train/validation/test %
            <input style={{ width: 90 }} value={ratios} onChange={(ev) => setRatios(ev.target.value)} />
          </label>
          <label>Date block (days)
            <input type="number" min={1} max={366} style={{ width: 60 }} value={blockDays} onChange={(ev) => setBlockDays(Number(ev.target.value))} />
          </label>
          <button className="primary" disabled={!name.trim() || e.eligible === 0} onClick={() => act(() => {
            const r = ratios.split("/").map((x) => Number(x) / 100);
            return post(`/missions/${missionId}/learning/datasets`, { name, ratios: r, block_days: blockDays });
          }, (r) => { setName(""); return `Dataset version “${r.name}” frozen: ${r.stats.images} images in ${r.stats.groups} groups.`; })}>Freeze a new version</button>
        </div>
        <p className="muted">Images are split by camera view and date block, so frames of one view on one day never sit on both
          sides. Groups that were once in a test set stay in test in every later version.</p>
        {s.datasets.length === 0 ? <div className="muted">No dataset versions yet.</div> : (
          <table className="t">
            <thead><tr><th>#</th><th>Name</th><th>Images</th><th>Train</th><th>Validation</th><th>Test</th><th>Duplicates dropped</th><th>By</th><th>Created (UTC)</th><th></th></tr></thead>
            <tbody>{s.datasets.map((d: any) => {
              const sp = d.stats.splits ?? {};
              const cell = (k: string) => sp[k] ? `${sp[k].images} img · ${sp[k].objects} obj · ${sp[k].groups} grp` : "–";
              return (
                <tr key={d.id}>
                  <td>{d.id}</td><td className="bidi">{d.name}</td><td>{d.stats.images}</td>
                  <td>{cell("train")}</td><td>{cell("val")}</td><td>{cell("test")}</td><td>{d.stats.duplicates_dropped}</td>
                  <td>{d.author}</td><td className="mono">{d.created_at}</td>
                  <td><button onClick={() => act(() => post(`/missions/${missionId}/learning/datasets/${d.id}/export`),
                    (r) => `Exported to ${r.folder}`)}>Export for training</button></td>
                </tr>);
            })}</tbody>
          </table>)}
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Evaluate a model</h2>
        <div className="row">
          <select value={evalDs} onChange={(ev) => setEvalDs(ev.target.value ? Number(ev.target.value) : "")} aria-label="Dataset version">
            <option value="">Dataset version…</option>{s.datasets.map((d: any) => <option key={d.id} value={d.id}>{d.id}: {d.name}</option>)}
          </select>
          <select value={evalSplit} onChange={(ev) => setEvalSplit(ev.target.value)} aria-label="Split">
            <option value="test">test</option><option value="val">validation</option><option value="train">train</option>
          </select>
          <select value={evalModel} onChange={(ev) => setEvalModel(ev.target.value)} aria-label="Model">
            <option value="">Model in production use</option>
            {s.models.map((md: any) => <option key={`${md.name}/${md.version}`} value={`${md.name}/${md.version}`}>{md.name} {md.version}{md.evaluation_only ? " (evaluation only)" : ""}</option>)}
          </select>
          <button className="primary" disabled={evalDs === "" || evaluating} onClick={() => act(() => {
            const [n, v] = evalModel ? evalModel.split("/") : [null, null];
            return post(`/missions/${missionId}/learning/evaluate`, { dataset_id: evalDs, split: evalSplit, name: n, version: v });
          }, () => "Evaluation started.")}>Evaluate</button>
          {evaluating && <span className="muted">Evaluating: {job.done} of {job.total || "?"} images…</span>}
        </div>
        {s.evaluations.length > 0 && (
          <table className="t" style={{ marginTop: 8 }}>
            <thead><tr><th>#</th><th>Dataset</th><th>Split</th><th>Model</th><th>State</th><th>Precision</th><th>Recall</th><th>F1</th><th>AP50</th><th>Count error</th><th></th></tr></thead>
            <tbody>{s.evaluations.map((ev: any) => {
              const o = ev.report?.overall;
              const md = ev.model?.model ?? {};
              return (<Fragment key={ev.id}>
                <tr>
                  <td>{ev.id}</td><td>{ev.dataset_id}</td><td>{ev.split}</td><td>{md.name} {md.version}</td>
                  <td>{ev.state}{ev.error ? `: ${ev.error}` : ""}</td>
                  <td>{pct(o?.precision)}</td><td>{pct(o?.recall)}</td><td>{pct(o?.f1)}</td><td>{pct(o?.ap50)}</td><td>{num(o?.count_mae)}</td>
                  <td className="row" style={{ gap: 4 }}>
                    {o && <button onClick={() => setOpen(open === ev.id ? null : ev.id)}>{open === ev.id ? "Hide" : "Details"}</button>}
                    {ev.state === "completed" && md.name && (s.in_use?.name !== md.name || s.in_use?.version !== md.version) &&
                      <button onClick={() => {
                        const note = window.prompt(`Promote ${md.name} ${md.version} to production use? It will process every new image. Optional note:`, "");
                        if (note === null) return;
                        act(() => post("/models/promote", { name: md.name, version: md.version, mission_id: missionId, evaluation_id: ev.id, note }),
                          () => `${md.name} ${md.version} is now the vehicle detector in production use.`);
                      }}>Promote…</button>}
                  </td>
                </tr>
                {open === ev.id && o && <tr><td colSpan={11}><EvalDetail report={ev.report} /></td></tr>}
              </Fragment>);
            })}</tbody>
          </table>)}
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Vehicle detector in production use</h2>
        <p style={{ margin: "4px 0" }}>
          {prod ? <>{prod.name} {prod.version}, promoted by {prod.author} on {prod.created_at} with evaluation {prod.evaluation_id} of mission {prod.mission_id}.</>
            : `The built-in default${s.in_use ? ` (${s.in_use.name} ${s.in_use.version})` : ""}. No model has been promoted.`}
        </p>
        <div className="row">
          <button className="danger" disabled={!prod} onClick={() => act(() => post("/models/rollback", {}),
            (r) => r.production ? `Back to ${r.production.name} ${r.production.version}.` : "Back to the built-in default.")}>Roll back</button>
          <span className="muted">Only an administrator can promote or roll back. Images already processed keep the results of the model that processed them.</span>
        </div>
        {s.history.length > 0 && <ul className="muted" style={{ marginBottom: 0 }}>{s.history.map((h: any) =>
          <li key={h.id}>{h.created_at}: {h.action === "promote" ? "promoted" : "rolled back from"} {h.name} {h.version} by {h.author}{h.note ? ` (${h.note})` : ""}</li>)}</ul>}
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Training</h2>
        <p className="muted">No algorithm trains in this version. Export a dataset version and train on a GPU machine; bring the new model
          back as a model pack, evaluate it here and promote it if it is better. Planned adapters:</p>
        <table className="t"><tbody>{s.adapters.map((a: any) => <tr key={a.algorithm}><td>{a.algorithm}</td><td>{a.adapter}</td><td>{a.status.replace("_", " ")}</td></tr>)}</tbody></table>
      </div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
    </div>
  );
}

function EvalDetail({ report }: { report: any }) {
  const rows = (title: string, groups: Record<string, any>) => Object.entries(groups).map(([k, o]) => (
    <tr key={title + k}><td>{title}</td><td>{k}</td><td>{o.images} ({o.full_images} full)</td><td>{o.references}</td>
      <td>{pct(o.precision)}</td><td>{pct(o.recall)}</td><td>{pct(o.ap50)}</td><td>{num(o.count_mae)}</td><td>{num(o.count_bias)}</td><td>{pct(o.class_agreement)}</td></tr>));
  return (
    <div>
      <table className="t">
        <thead><tr><th></th><th></th><th>Images</th><th>Vehicles</th><th>Precision</th><th>Recall</th><th>AP50</th><th>Count error</th><th>Count bias</th><th>Class agrees</th></tr></thead>
        <tbody>{rows("All", { all: report.overall })}{rows("Light", report.by_condition)}{rows("View", report.by_view)}</tbody>
      </table>
      <ul className="muted">{report.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}<li>Match rule: IoU at least {report.iou}.</li></ul>
    </div>
  );
}
