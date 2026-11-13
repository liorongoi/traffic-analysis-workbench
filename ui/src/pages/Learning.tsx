import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Choice, NamePick, thisMonth, today } from "../components/Pick";
import { api, ApiError, fmtDuration, post } from "../api";
import { CompareBars, LossChart } from "../components/Charts";

// Learning: which reviewed images can teach a model, frozen dataset versions, training a new vehicle detector on this
// computer, evaluation of a model on them, and the operator's promotion of a model to production use. Nothing is
// promoted on its own.

const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
const num = (v: number | null | undefined, d = 2) => (v == null ? "–" : v.toFixed(d));

type Ref = { mission_id: string; dataset_id: number };
const refKey = (r: Ref) => `${r.mission_id}:${r.dataset_id}`;
type Act = (fn: () => Promise<any>, ok?: (r: any) => string) => Promise<void>;

/** The dataset versions chosen by default: the newest of this mission, plus the data of rounds that did not improve. */
function defaultRefs(s: any, missionId: string): Ref[] {
  const out: Ref[] = [...(s.carry_over?.datasets ?? [])];
  const newest = s.datasets[0];
  if (newest && !out.some((r) => r.mission_id === missionId && r.dataset_id === newest.id)) out.push({ mission_id: missionId, dataset_id: newest.id });
  return out;
}

function dsLabel(s: any, missionId: string, r: Ref): string {
  const d = (s.all_datasets as any[]).find((x) => x.mission_id === r.mission_id && x.id === r.dataset_id);
  if (!d) return `${r.mission_id === missionId ? "" : "another mission: "}#${r.dataset_id}`;
  return r.mission_id === missionId ? `${d.id}: ${d.name}` : `${d.mission} · ${d.id}: ${d.name}`;
}

/** Choose one or more dataset versions, from this mission or any other still-image mission. */
function DatasetPicker({ s, missionId, value, onChange, split = "train", label = "Dataset versions" }: {
  s: any; missionId: string; value: Ref[]; onChange: (v: Ref[]) => void; split?: string; label?: string;
}) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  const groups = useMemo(() => {
    const g = new Map<string, { name: string; items: any[] }>();
    for (const d of s.all_datasets as any[]) {
      const k = d.mission_id;
      if (!g.has(k)) g.set(k, { name: k === missionId ? "This mission" : d.mission, items: [] });
      g.get(k)!.items.push(d);
    }
    return [...g.entries()];
  }, [s.all_datasets, missionId]);
  const keys = new Set(value.map(refKey));
  const toggle = (r: Ref) => onChange(keys.has(refKey(r)) ? value.filter((x) => refKey(x) !== refKey(r)) : [...value, r]);
  const summary = value.length === 0 ? `${label}…` : value.length === 1 ? dsLabel(s, missionId, value[0]) : `${value.length} dataset versions`;
  return (
    <div className="dspick" ref={box}>
      <button aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen(!open)} title={value.map((r) => dsLabel(s, missionId, r)).join("\n")}>
        {summary} <span aria-hidden>▾</span></button>
      {open && <div className="dspick-menu" role="listbox" aria-multiselectable aria-label={label}>
        {groups.length === 0 && <div className="muted" style={{ padding: 6 }}>No dataset versions yet. Freeze one above.</div>}
        {groups.map(([mid, g]) => <div key={mid}>
          <h4>{g.name}</h4>
          {g.items.map((d) => {
            const r = { mission_id: mid, dataset_id: d.id };
            const sp = d.stats?.splits?.[split];
            return <label key={refKey(r)}>
              <input type="checkbox" checked={keys.has(refKey(r))} onChange={() => toggle(r)} />
              <span><span className="bidi">{d.id}: {d.name}</span>
                <small>{d.stats?.images ?? 0} images · {sp?.images ?? 0} {split === "val" ? "validation" : split} · by {d.author}, {d.created_at?.slice(0, 10)}</small></span>
            </label>;
          })}
        </div>)}
        <p className="muted small" style={{ margin: "8px 6px 2px", fontSize: 12 }}>Chosen versions are used together. An image in a test set in
          any of them stays in test, and identical images count once.</p>
      </div>}
    </div>
  );
}

const RATIOS: [string, string][] = [["70/15/15", "70 / 15 / 15"], ["80/10/10", "80 / 10 / 10"], ["60/20/20", "60 / 20 / 20"],
  ["70/20/10", "70 / 20 / 10"], ["75/15/10", "75 / 15 / 10"], ["50/25/25", "50 / 25 / 25"]];
const BLOCK_DAYS: [number, string][] = [[1, "1 day"], [2, "2 days"], [3, "3 days"], [7, "1 week"], [14, "2 weeks"], [30, "30 days"], [90, "90 days"]];
const EPOCHS = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300];

export default function LearningTab({ missionId, job, reload }: { missionId: string; job: any; reload: () => void }) {
  const [s, setS] = useState<any>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [name, setName] = useState(`Version of ${today()}`);
  const [ratios, setRatios] = useState("70/15/15");
  const [blockDays, setBlockDays] = useState(1);
  const [evalRefs, setEvalRefs] = useState<Ref[] | null>(null);
  const [evalSplit, setEvalSplit] = useState("test");
  const [evalModel, setEvalModel] = useState("");
  const [open, setOpen] = useState<number | null>(null);
  const [cmp, setCmp] = useState<{ cur: number | null; prev: number | null }>({ cur: null, prev: null });

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
  const eRefs = evalRefs ?? defaultRefs(s, missionId);
  const savedByPack = new Map<string, any>((s.saved_models as any[]).map((x) => [`${x.model_name}/${x.model_version}`, x]));
  const modelLabel = (md: any) => { const sv = savedByPack.get(`${md?.name}/${md?.version}`); return sv ? `${sv.name} (${md.name} ${md.version})` : `${md?.name ?? "?"} ${md?.version ?? ""}`; };
  const dataLabel = (ev: any) => (ev.sources as Ref[]).map((r) => dsLabel(s, missionId, r)).join(" + ");
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
          <label>Version name
            <NamePick label="Version name" value={name} onChange={setName}
              suggestions={[`Version of ${today()}`, `${thisMonth()}, day`, `${thisMonth()}, night`, `${thisMonth()}, day and night`, `Version ${s.datasets.length + 1}`]} />
          </label>
          <label>Train/validation/test %
            <Choice label="Train/validation/test %" value={ratios} onChange={setRatios} options={RATIOS} />
          </label>
          <label>Date block
            <Choice label="Date block (days)" value={blockDays} onChange={setBlockDays} options={BLOCK_DAYS} />
          </label>
          <button className="primary" disabled={!name.trim() || e.eligible === 0} onClick={() => act(() => {
            const r = ratios.split("/").map((x) => Number(x) / 100);
            return post(`/missions/${missionId}/learning/datasets`, { name, ratios: r, block_days: blockDays });
          }, (r) => { setName(`Version of ${today()}`); return `Dataset version “${r.name}” frozen: ${r.stats.images} images in ${r.stats.groups} groups.`; })}>Freeze a new version</button>
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
          <DatasetPicker s={s} missionId={missionId} value={eRefs} onChange={setEvalRefs} split={evalSplit} />
          <select value={evalSplit} onChange={(ev) => setEvalSplit(ev.target.value)} aria-label="Split">
            <option value="test">test</option><option value="val">validation</option><option value="train">train</option>
          </select>
          <select value={evalModel} onChange={(ev) => setEvalModel(ev.target.value)} aria-label="Model">
            <option value="">Model in production use</option>
            {s.saved_models.length > 0 && <optgroup label="Saved models">
              {s.saved_models.filter((x: any) => x.installed).map((x: any) => <option key={x.id} value={`${x.model_name}/${x.model_version}`}>{x.name}</option>)}
            </optgroup>}
            <optgroup label="All installed models">
              {s.models.map((md: any) => <option key={`${md.name}/${md.version}`} value={`${md.name}/${md.version}`}>{md.name} {md.version}{md.evaluation_only ? " (evaluation only)" : ""}</option>)}
            </optgroup>
          </select>
          <button className="primary" disabled={eRefs.length === 0 || evaluating} onClick={() => act(() => {
            const [n, v] = evalModel ? evalModel.split("/") : [null, null];
            return post(`/missions/${missionId}/learning/evaluate`, { datasets: eRefs, split: evalSplit, name: n, version: v });
          }, () => "Evaluation started.")}>Evaluate</button>
          {evaluating && <span className="muted">Evaluating: {job.done} of {job.total || "?"} images…</span>}
        </div>
        {(s.carry_over?.runs?.length ?? 0) > 0 && evalRefs === null && <p className="muted" style={{ margin: "8px 0 0" }}>
          Chosen for you: the newest dataset version and the data of {s.carry_over.runs.length === 1 ? "the training round" : "the training rounds"}{" "}
          {s.carry_over.runs.map((r: number) => `#${r}`).join(", ")} that did not improve on the model {s.carry_over.runs.length === 1 ? "it" : "they"} started from.</p>}
        {s.evaluations.length > 0 && (
          <table className="t" style={{ marginTop: 8 }}>
            <thead><tr><th>#</th><th>Data</th><th>Split</th><th>Model</th><th>State</th><th>Precision</th><th>Recall</th><th>F1</th><th>AP50</th><th>Count error</th><th></th></tr></thead>
            <tbody>{s.evaluations.map((ev: any) => {
              const o = ev.report?.overall;
              const md = ev.model?.model ?? {};
              return (<Fragment key={ev.id}>
                <tr>
                  <td>{ev.id}</td><td className="bidi" style={{ maxWidth: 220, whiteSpace: "normal" }}>{dataLabel(ev)}</td><td>{ev.split}</td><td className="bidi">{modelLabel(md)}</td>
                  <td>{ev.state}{ev.error ? `: ${ev.error}` : ""}</td>
                  <td>{pct(o?.precision)}</td><td>{pct(o?.recall)}</td><td>{pct(o?.f1)}</td><td>{pct(o?.ap50)}</td><td>{num(o?.count_mae)}</td>
                  <td className="row" style={{ gap: 4 }}>
                    {o && <button onClick={() => setOpen(open === ev.id ? null : ev.id)}>{open === ev.id ? "Hide" : "Details"}</button>}
                    {o && <button onClick={() => setCmp({ cur: ev.id, prev: null })} title="Compare with the previous evaluation on the same images">Compare</button>}
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
        {s.evaluations.filter((x: any) => x.state === "completed").length >= 2 &&
          <EvalCompare evals={s.evaluations} cmp={cmp} setCmp={setCmp} modelLabel={modelLabel} dataLabel={dataLabel} />}
      </div>

      <RoundsCard s={s} missionId={missionId} act={act} modelLabel={modelLabel} dataLabel={(r: Ref[]) => r.map((x) => dsLabel(s, missionId, x)).join(" + ")} />

      <SavedModelsCard s={s} missionId={missionId} act={act} />

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

function TrainCard({ missionId, s, act }: { missionId: string; s: any; act: Act }) {
  const t = s.training ?? { state: "checking" };
  const job = t.job;
  const active = job && ["running", "paused"].includes(job.state);
  const trainable = (s.models as any[]).filter((md) => md.trainable);
  const prodKey = s.production ? `${s.production.name}/${s.production.version}` : "";
  const [refs, setRefs] = useState<Ref[] | null>(null);
  const ds = refs ?? defaultRefs(s, missionId);
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
          <DatasetPicker s={s} missionId={missionId} value={ds} onChange={setRefs} label="Dataset versions to train on" />
          <label>Start from
            <select value={startFrom} onChange={(ev) => setStart(ev.target.value)}>
              {t.coco_weights && <option value="coco">the original detector (YOLOX-s)</option>}
              {trainable.map((md: any) => <option key={`${md.name}/${md.version}`} value={`${md.name}/${md.version}`}>{md.saved_as ? `${md.saved_as} (${md.name} ${md.version})` : `${md.name} ${md.version}`}{md.trained_on ? `, trained on “${md.trained_on}”` : ""}{`${md.name}/${md.version}` === prodKey ? " (in use)" : ""}</option>)}
            </select>
          </label>
          <label>Epochs
            <Choice label="Epochs" value={ep} onChange={setEpochs}
              options={[...new Set([...EPOCHS, t.suggested_epochs ?? 20])].sort((a, b) => a - b).map((n): [number, string] => [n, n === (t.suggested_epochs ?? 20) ? `${n} (suggested)` : String(n)])} />
          </label>
          <button className="primary" disabled={ds.length === 0 || !startFrom || !!t.busy_mission} onClick={() => act(
            () => post(`/missions/${missionId}/learning/train`, { datasets: ds, epochs: ep, start_from: startFrom }),
            () => "Training started. You can close the browser; it continues on this computer.")}>Train</button>
        </div>}
        {!active && (s.carry_over?.runs?.length ?? 0) > 0 && refs === null && <div className="msg warn">
          {s.carry_over.runs.length === 1 ? "The last training round" : `The last ${s.carry_over.runs.length} training rounds`}{" "}
          ({s.carry_over.runs.map((r: number) => `#${r}`).join(", ")}) did not improve on the model they started from, so their data is
          chosen again together with the newest dataset version: {ds.map((r) => dsLabel(s, missionId, r)).join(", ")}. Change it with the
          dataset button.</div>}
        {!active && t.busy_mission && t.busy_mission !== missionId && <p className="muted">Another mission is training a model on this computer; one training runs at a time.</p>}
        <p className="muted" style={{ marginBottom: 0 }}>Each epoch goes once through the training images; the epoch with the lowest validation
          loss is kept. The new model is then added and evaluated on the test images, beside the model it started from, so you
          can see whether it improved. If it did not, its data is offered again for the next round. The model in use does
          not change until you promote or save the new one.</p>
      </>}
      {job && <div style={{ marginTop: 10 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <b>{job.state === "paused" ? "Paused" : job.state === "running" ? job.stage : job.state === "finished" ? "Training finished" : `Training ${job.state}`}</b>
          {active && job.total > 0 && <span className="muted">{job.done} of {job.total} steps{r.epochs ? ` · epoch ${Math.min((r.epoch ?? 0) + 1, r.epochs)} of ${r.epochs}` : ""}</span>}
        </div>
        {active && <div className="progress" style={{ margin: "6px 0" }}><div style={{ width: `${pct}%` }} /></div>}
        {active && <div className="muted">Elapsed {fmtDuration(job.elapsed_s)} · remaining {job.eta_s != null ? `about ${fmtDuration(job.eta_s)}` : "–"} ({job.eta_basis}){r.device ? ` · on ${r.device}` : ""}</div>}
        {job.state === "finished" && r.model && <div className="msg ok">New model {r.model.name} {r.model.version} added
          {r.evaluation ? `, and evaluated as evaluation ${r.evaluation} below. Compare it with the model in use and promote it if it is better.`
            : r.evaluation_error ? `. Evaluating it failed (${r.evaluation_error}); evaluate it below by hand.` : ". Evaluate it below, then promote it if it is better."}</div>}
        {job.state === "finished" && r.improved === true && <div className="msg ok">It is better than the model it started from on the same test images. Promote it, or save it by name below.</div>}
        {job.state === "finished" && r.improved === false && <div className="msg warn">It did not improve on the model it started from. Its data is kept for the next round: add more reviewed images, freeze a new dataset version and train again.</div>}
        {job.error && <div className="msg err">{job.error}</div>}
        {job.state === "cancelled" && <div className="muted">Stopped; no model was added.</div>}
        {active && <div className="row" style={{ marginTop: 6 }}>
          {job.state === "running" && <button onClick={() => control("pause")}>Pause</button>}
          {job.state === "paused" && <button className="primary" onClick={() => control("resume")}>Resume</button>}
          <button className="danger" onClick={() => { if (window.confirm("Stop training? What it learned so far is discarded.")) control("cancel"); }}>Stop training</button>
        </div>}
        {r.history?.length > 0 && <div style={{ marginTop: 10 }}>
          <LossChart history={r.history} />
          <details style={{ marginTop: 6 }}><summary className="muted">Loss per epoch as numbers</summary>
            <table className="t"><thead><tr><th>Epoch</th><th>Training loss</th><th>Validation loss</th></tr></thead>
              <tbody>{r.history.map((h: any) => <tr key={h.epoch}><td>{h.epoch}</td><td>{h.train_loss}</td><td>{h.val_loss ?? "–"}</td></tr>)}</tbody></table>
          </details></div>}
      </div>}
    </div>
  );
}

/** One evaluation beside another, as bars: overall measures, and F1 by light condition. */
function EvalCompare({ evals, cmp, setCmp, modelLabel, dataLabel }: {
  evals: any[]; cmp: { cur: number | null; prev: number | null }; setCmp: (c: { cur: number | null; prev: number | null }) => void;
  modelLabel: (md: any) => string; dataLabel: (ev: any) => string;
}) {
  const done = evals.filter((x) => x.state === "completed" && x.report);
  const cur = done.find((x) => x.id === cmp.cur) ?? done[0];
  const older = done.filter((x) => x.id < cur.id);
  const md = (x: any) => x.model?.model ?? {};
  const sameModel = (a: any, b: any) => md(a).name === md(b).name && md(a).version === md(b).version;
  // The previous one: the last evaluation of another model on exactly the same images, else the last on the same images.
  const auto = older.find((x) => x.data_key === cur.data_key && !sameModel(x, cur)) ?? older.find((x) => x.data_key === cur.data_key) ?? older[0];
  const prev = done.find((x) => x.id === cmp.prev) ?? auto;
  const label = (x: any) => `#${x.id} · ${modelLabel(md(x))}`;
  if (!prev) return null;
  const a = prev.report.overall, b = cur.report.overall;
  const conds = [...new Set([...Object.keys(prev.report.by_condition ?? {}), ...Object.keys(cur.report.by_condition ?? {})])]
    .filter((k) => k !== "unknown");
  const diff = (x: number | null | undefined, y: number | null | undefined, lowerBetter = false) => {
    if (x == null || y == null) return "–";
    const d = y - x;
    if (Math.abs(d) < 0.005) return <span className="muted">same</span>;
    const good = lowerBetter ? d < 0 : d > 0;
    return <span className={good ? "delta-up" : "delta-down"}>{d > 0 ? "+" : ""}{d.toFixed(2)} {good ? "better" : "worse"}</span>;
  };
  return (
    <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--line)" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>Compare evaluations</h3>
        <div className="row">
          <label>Current <select value={cur.id} onChange={(e) => setCmp({ cur: Number(e.target.value), prev: null })}>
            {done.map((x) => <option key={x.id} value={x.id}>{label(x)}</option>)}</select></label>
          <label>Previous <select value={prev.id} onChange={(e) => setCmp({ cur: cur.id, prev: Number(e.target.value) })}>
            {done.filter((x) => x.id !== cur.id).map((x) => <option key={x.id} value={x.id}>{label(x)}{x.data_key === cur.data_key ? "" : " (other images)"}</option>)}</select></label>
        </div>
      </div>
      {prev.data_key !== cur.data_key && <div className="msg warn">These two looked at different images ({dataLabel(prev)}, {prev.split}; and {dataLabel(cur)}, {cur.split}),
        so the difference is partly the images, not only the model.</div>}
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16, marginTop: 8 }}>
        <div>
          <div className="muted" style={{ fontSize: 12.5 }}>All images</div>
          <CompareBars prevName={label(prev)} curName={label(cur)} groups={[
            { label: "Precision", prev: a.precision, cur: b.precision }, { label: "Recall", prev: a.recall, cur: b.recall },
            { label: "F1", prev: a.f1, cur: b.f1 }, { label: "AP50", prev: a.ap50, cur: b.ap50 },
            { label: "Class agrees", prev: a.class_agreement, cur: b.class_agreement }]} />
        </div>
        {conds.length > 0 && <div>
          <div className="muted" style={{ fontSize: 12.5 }}>F1 by light</div>
          <CompareBars prevName={label(prev)} curName={label(cur)} groups={conds.map((k) => ({
            label: k[0].toUpperCase() + k.slice(1), prev: prev.report.by_condition?.[k]?.f1, cur: cur.report.by_condition?.[k]?.f1 }))} />
        </div>}
      </div>
      <table className="t" style={{ marginTop: 8 }}>
        <thead><tr><th></th><th>Previous</th><th>Current</th><th>Change</th></tr></thead>
        <tbody>
          <tr><td>Vehicles found (recall)</td><td>{a.true_positives} of {a.references}</td><td>{b.true_positives} of {b.references}</td><td>{a.recall == null || b.recall == null ? "–" : Math.abs(b.recall - a.recall) < 0.0005 ? <span className="muted">same</span>
            : <span className={b.recall > a.recall ? "delta-up" : "delta-down"}>{b.recall > a.recall ? "+" : ""}{((b.recall - a.recall) * 100).toFixed(1)} pts</span>}</td></tr>
          <tr><td>Wrong boxes on fully reviewed images</td><td>{a.false_positives_full}</td><td>{b.false_positives_full}</td><td>{diff(a.false_positives_full, b.false_positives_full, true)}</td></tr>
          <tr><td>Count error per image</td><td>{num(a.count_mae)}</td><td>{num(b.count_mae)}</td><td>{diff(a.count_mae, b.count_mae, true)}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

/** Every training round: what it trained on, whether it beat the model it started from, and its loss per epoch. */
function RoundsCard({ s, missionId, act, modelLabel, dataLabel }: {
  s: any; missionId: string; act: Act; modelLabel: (md: any) => string; dataLabel: (r: Ref[]) => string;
}) {
  const runs: any[] = s.training_runs ?? [];
  const [sel, setSel] = useState<number | null>(null);
  const [saving, setSaving] = useState<any | null>(null);
  if (!runs.length) return null;
  const ev = new Map<number, any>((s.evaluations as any[]).map((x) => [x.id, x]));
  const shown = runs.find((r) => r.id === sel) ?? runs.find((r) => r.history?.length);
  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Training rounds</h2>
      <table className="t">
        <thead><tr><th>Round</th><th>Trained on</th><th>Started from</th><th>Epochs</th><th>New model</th><th>F1: start → new</th><th>Result</th><th></th></tr></thead>
        <tbody>{runs.map((r) => {
          const n = ev.get(r.evaluation_id)?.report?.overall, b = ev.get(r.baseline_evaluation_id)?.report?.overall;
          const saved = (s.saved_models as any[]).find((x) => x.model_name === r.model_name && x.model_version === r.model_version);
          return <tr key={r.id} className={shown?.id === r.id ? "sel" : ""}>
            <td>#{r.id}</td><td className="bidi" style={{ maxWidth: 240, whiteSpace: "normal" }}>{dataLabel(r.sources)}</td>
            <td className="bidi">{r.start_from === "coco" ? "YOLOX-s" : modelLabel({ name: r.start_from.split("/")[0], version: r.start_from.split("/")[1] })}</td>
            <td>{r.epochs}</td><td className="bidi">{r.model_name ? modelLabel({ name: r.model_name, version: r.model_version }) : "–"}</td>
            <td>{b || n ? `${pct(b?.f1)} → ${pct(n?.f1)}` : "–"}</td>
            <td className="round-result">{r.state !== "finished" ? <span className={`badge ${r.state === "failed" ? "failed" : ""}`}>{r.state}</span>
              : r.improved === true ? <span className="badge ok">▲ improved</span>
              : r.improved === false ? <span className="badge failed">did not improve</span> : <span className="badge">not compared</span>}
              {saved && <> <span className="badge">saved as {saved.name}</span></>}</td>
            <td className="row" style={{ gap: 4 }}>
              {r.history?.length > 0 && <button onClick={() => setSel(r.id)} disabled={shown?.id === r.id}>Loss graph</button>}
              {r.model_name && !saved && <button onClick={() => setSaving(r)}>Save model…</button>}
            </td>
          </tr>;
        })}</tbody>
      </table>
      {saving && <SaveForm s={s} missionId={missionId} act={act} model={{ name: saving.model_name, version: saving.model_version }}
        evaluationId={saving.evaluation_id} onDone={() => setSaving(null)} />}
      {shown?.history?.length > 0 && <div style={{ marginTop: 12 }}>
        <div className="muted" style={{ fontSize: 12.5, marginBottom: 4 }}>Round #{shown.id}: loss per epoch</div>
        <LossChart history={shown.history} />
      </div>}
    </div>
  );
}

const MODEL_USES = ["Daytime images", "Night images", "Day and night images", "Fixed cameras", "Moving cameras",
  "Highway traffic", "City junctions", "Trucks and buses", "Testing only"];

function SaveForm({ s, missionId, act, model, evaluationId, onDone }: {
  s: any; missionId: string; act: Act; model?: { name: string; version: string } | null; evaluationId?: number | null; onDone?: () => void;
}) {
  const unsaved = (s.models as any[]).filter((md) => !md.saved_as && !md.evaluation_only);
  const [pick, setPick] = useState(model ? `${model.name}/${model.version}` : (unsaved.length ? `${unsaved[unsaved.length - 1].name}/${unsaved[unsaved.length - 1].version}` : ""));
  const [n, v] = pick.split("/");
  const trainedOn = (s.models as any[]).find((md) => md.name === n && md.version === v)?.trained_on as string | undefined;
  const nameIdeas = [trainedOn ? `${trainedOn} detector` : "", `Vehicles, ${thisMonth()}`, `Day vehicles, ${thisMonth()}`,
    `Night vehicles, ${thisMonth()}`, `Day and night, ${thisMonth()}`, n && v ? `${n} ${v}` : ""].filter(Boolean);
  const [name, setName] = useState(nameIdeas[0]);
  const [desc, setDesc] = useState("");
  const ev = evaluationId ?? (s.evaluations as any[]).find((x) => x.state === "completed" && x.model?.model?.name === n && x.model?.model?.version === v)?.id;
  return (
    <div className="card" style={{ background: "var(--panel-2)", marginTop: 10 }}>
      <div className="row">
        {!model && <select value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Model to save">
          <option value="">Model…</option>
          {unsaved.map((md) => <option key={`${md.name}/${md.version}`} value={`${md.name}/${md.version}`}>{md.name} {md.version}{md.trained_on ? `, trained on “${md.trained_on}”` : ""}</option>)}
        </select>}
        {model && <span className="mono">{model.name} {model.version}</span>}
        <NamePick label="Name" value={name} onChange={setName} suggestions={nameIdeas} />
        <NamePick optional label="What it is for" value={desc} onChange={setDesc} suggestions={MODEL_USES} />
        <button className="primary" disabled={!pick || !name.trim()} onClick={() => act(
          () => post("/models/saved", { name, description: desc || null, model_name: n, model_version: v, mission_id: ev ? missionId : null, evaluation_id: ev ?? null }),
          (r) => { onDone?.(); setDesc(""); return `Saved and frozen as “${r.name}”. Any mission can now use it.`; })}>Save and freeze</button>
        {onDone && <button onClick={onDone}>Cancel</button>}
      </div>
      <p className="muted" style={{ margin: "8px 0 0", fontSize: 12.5 }}>Saving freezes the model: its files become read-only and are checked
        every time it is used. Training from it later makes a new model and never changes this one.</p>
    </div>
  );
}

/** Models kept by name, and which one this mission uses to find vehicles. */
function SavedModelsCard({ s, missionId, act }: { s: any; missionId: string; act: Act }) {
  const [adding, setAdding] = useState(false);
  const saved: any[] = s.saved_models ?? [];
  const use = (name: string | null) => act(() => api(`/missions/${missionId}/vehicle-model`, { method: "PUT", body: { saved: name } }),
    () => name ? `This mission now finds vehicles with “${name}” in new images.` : "This mission now uses the model in production use.");
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>Saved models</h2>
        {!adding && <button onClick={() => setAdding(true)}>Save a model…</button>}
      </div>
      <p className="muted">When a model is good enough and you do not need to keep training it, save it by name. Any mission can then
        process its images with it, so you can keep one model for cars at night, another for a busy junction, and so on.
        This mission uses: <b>{s.mission_model ?? "the model in production use"}</b>.</p>
      {adding && <SaveForm s={s} missionId={missionId} act={act} onDone={() => setAdding(false)} />}
      {saved.length === 0 ? <div className="muted">No saved models yet.</div> : (
        <div className="savedgrid" style={{ marginTop: 10 }}>
          {saved.map((x) => {
            const o = x.summary?.overall;
            const mine = s.mission_model && s.mission_model.toLowerCase() === x.name.toLowerCase();
            return <div key={x.id} className="card">
              <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
                <h3 className="bidi">{x.name}</h3>
                <span className="badge" title="Read-only files, checked by checksum every time it is used">🔒 frozen</span>
              </div>
              {x.description && <div className="bidi" style={{ fontSize: 13, marginBottom: 4 }}>{x.description}</div>}
              <div className="muted" style={{ fontSize: 12.5 }}>{x.model_name} {x.model_version}{x.summary?.trained_on ? ` · trained on “${x.summary.trained_on}”` : ""}</div>
              {o && <div style={{ fontSize: 12.5, marginTop: 4 }}>F1 {pct(o.f1)} · precision {pct(o.precision)} · recall {pct(o.recall)} · AP50 {pct(o.ap50)}</div>}
              <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>Saved by {x.author}, {x.created_at.slice(0, 10)}</div>
              {!x.installed && <div className="msg err" style={{ margin: "6px 0 0" }}>Its files are missing or changed; it cannot be used.</div>}
              {x.history?.length > 0 && <details style={{ marginTop: 6 }}><summary className="muted" style={{ fontSize: 12.5 }}>Loss per epoch</summary>
                <LossChart history={x.history} height={180} /></details>}
              <div className="row" style={{ marginTop: 8 }}>
                {mine ? <><span className="badge ok">Used by this mission</span><button onClick={() => use(null)}>Stop using</button></>
                  : <button className="primary" disabled={!x.installed} onClick={() => use(x.name)}>Use for this mission</button>}
                <button className="danger" onClick={() => {
                  if (window.confirm(`Take “${x.name}” off the saved list? Its files stay on this computer.`))
                    act(() => api(`/models/saved/${encodeURIComponent(x.name)}`, { method: "DELETE" }), () => `“${x.name}” is no longer saved.`);
                }}>Remove</button>
              </div>
            </div>;
          })}
        </div>)}
    </div>
  );
}
