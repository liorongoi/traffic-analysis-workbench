import { Fragment, useCallback, useEffect, useState } from "react";
import { api, ApiError, fmtDuration, post } from "../api";

// Learning: which reviewed images can teach a model, frozen dataset versions, training a new vehicle detector on this
// computer, evaluation of a model on them, and the operator's promotion of a model to production use. Nothing is
// promoted on its own.

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
  // While a model trains (or the computer is still being checked), refresh the page's figures.
  const tj = s?.training?.job;
  const training = tj && ["running", "paused"].includes(tj.state);
  const checking = s?.training?.state === "checking";
  useEffect(() => {
    if (!training && !checking) return;
    const t = setInterval(() => { load().catch(() => undefined); }, training ? 3000 : 1500);
    return () => clearInterval(t);
  }, [training, checking, load]);
  const trainState = tj?.state;
  useEffect(() => { if (trainState && !["running", "paused"].includes(trainState)) reload(); }, [trainState, reload]);
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
                    (r) => `Exported to ${r.folder}`)} title="A copy of the images and labels, for training on another computer with trafficwb-train">Export</button></td>
                </tr>);
            })}</tbody>
          </table>)}
      </div>

      <TrainCard missionId={missionId} s={s} act={act} />

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
        <h2 style={{ marginTop: 0 }}>What can learn</h2>
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

function TrainCard({ missionId, s, act }: { missionId: string; s: any; act: (fn: () => Promise<any>, ok?: (r: any) => string) => Promise<void> }) {
  const t = s.training ?? { state: "checking" };
  const job = t.job;
  const active = job && ["running", "paused"].includes(job.state);
  const trainable = (s.models as any[]).filter((md) => md.trainable);
  const prodKey = s.production ? `${s.production.name}/${s.production.version}` : "";
  const [ds, setDs] = useState<number | "">("");
  const [start, setStart] = useState<string>("");
  const [epochs, setEpochs] = useState<number | "">("");
  const startFrom = start || (trainable.some((md) => `${md.name}/${md.version}` === prodKey) ? prodKey : (t.coco_weights ? "coco" : (trainable[0] ? `${trainable[0].name}/${trainable[0].version}` : "")));
  const ep = epochs === "" ? (t.suggested_epochs ?? 20) : epochs;
  const control = (a: string) => act(() => post(`/missions/${missionId}/learning/train/${a}`));
  const pct = job && job.total ? Math.round((100 * job.done) / job.total) : 0;
  const r = job?.result ?? {};
  const cpu = t.device_type === "cpu";
  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Train a new vehicle detector here</h2>
      {t.state === "checking" && <p className="muted">Checking what this computer can train on…</p>}
      {t.state === "unavailable" && <div className="msg err">{t.reason}</div>}
      {t.state === "ready" && <>
        <p style={{ marginTop: 0 }}>Training runs on this computer: <b>{t.device}</b>.{" "}
          {cpu ? "On a processor without a supported GPU it is slow: for a few hundred images, expect a night. Processing of new images continues meanwhile, more slowly."
            : "For a few hundred images, expect roughly one to two hours."}</p>
        {!active && <div className="row">
          <select value={ds} onChange={(ev) => setDs(ev.target.value ? Number(ev.target.value) : "")} aria-label="Dataset version to train on">
            <option value="">Dataset version…</option>{s.datasets.map((d: any) => <option key={d.id} value={d.id}>{d.id}: {d.name} ({d.stats.splits?.train?.images ?? 0} training images)</option>)}
          </select>
          <label>Start from
            <select value={startFrom} onChange={(ev) => setStart(ev.target.value)}>
              {t.coco_weights && <option value="coco">the original detector (YOLOX-s)</option>}
              {trainable.map((md: any) => <option key={`${md.name}/${md.version}`} value={`${md.name}/${md.version}`}>{md.name} {md.version}{md.trained_on ? `, trained on “${md.trained_on}”` : ""}{`${md.name}/${md.version}` === prodKey ? " (in use)" : ""}</option>)}
            </select>
          </label>
          <label>Rounds
            <input type="number" min={1} max={300} style={{ width: 64 }} value={ep} onChange={(ev) => setEpochs(ev.target.value ? Number(ev.target.value) : "")} />
          </label>
          <button className="primary" disabled={ds === "" || !startFrom || !!t.busy_mission} onClick={() => act(
            () => post(`/missions/${missionId}/learning/train`, { dataset_id: ds, epochs: ep, start_from: startFrom }),
            () => "Training started. You can close the browser; it continues on this computer.")}>Train</button>
        </div>}
        {!active && t.busy_mission && t.busy_mission !== missionId && <p className="muted">Another mission is training a model on this computer; one training runs at a time.</p>}
        <p className="muted" style={{ marginBottom: 0 }}>Each round goes once through the training images; the round with the lowest validation
          loss is kept. The new model is then added and evaluated on the dataset's test images below. The model in use does
          not change until you promote the new one.</p>
      </>}
      {job && <div style={{ marginTop: 10 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <b>{job.state === "paused" ? "Paused" : job.state === "running" ? job.stage : job.state === "finished" ? "Training finished" : `Training ${job.state}`}</b>
          {active && job.total > 0 && <span className="muted">{job.done} of {job.total} steps{r.epochs ? ` · round ${Math.min((r.epoch ?? 0) + 1, r.epochs)} of ${r.epochs}` : ""}</span>}
        </div>
        {active && <div className="progress" style={{ margin: "6px 0" }}><div style={{ width: `${pct}%` }} /></div>}
        {active && <div className="muted">Elapsed {fmtDuration(job.elapsed_s)} · remaining {job.eta_s != null ? `about ${fmtDuration(job.eta_s)}` : "–"} ({job.eta_basis}){r.device ? ` · on ${r.device}` : ""}</div>}
        {job.state === "finished" && r.model && <div className="msg ok">New model {r.model.name} {r.model.version} added
          {r.evaluation ? `, and evaluated as evaluation ${r.evaluation} below. Compare it with the model in use and promote it if it is better.`
            : r.evaluation_error ? `. Evaluating it failed (${r.evaluation_error}); evaluate it below by hand.` : ". Evaluate it below, then promote it if it is better."}</div>}
        {job.error && <div className="msg err">{job.error}</div>}
        {job.state === "cancelled" && <div className="muted">Stopped; no model was added.</div>}
        {active && <div className="row" style={{ marginTop: 6 }}>
          {job.state === "running" && <button onClick={() => control("pause")}>Pause</button>}
          {job.state === "paused" && <button className="primary" onClick={() => control("resume")}>Resume</button>}
          <button className="danger" onClick={() => { if (window.confirm("Stop training? What it learned so far is discarded.")) control("cancel"); }}>Stop training</button>
        </div>}
        {r.history?.length > 0 && <details style={{ marginTop: 6 }}><summary className="muted">Rounds so far</summary>
          <table className="t"><thead><tr><th>Round</th><th>Training loss</th><th>Validation loss</th></tr></thead>
            <tbody>{r.history.map((h: any) => <tr key={h.epoch}><td>{h.epoch}</td><td>{h.train_loss}</td><td>{h.val_loss ?? "–"}</td></tr>)}</tbody></table>
        </details>}
      </div>}
    </div>
  );
}
