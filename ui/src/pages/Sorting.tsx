import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError, fmtDuration, post } from "../api";
import "../sorting.css";

// Sorting a bundle of still images into indoor and outdoor. Indoor: the camera is inside a place and shows the inside.
// Outdoor: the camera is outside, or it is inside and the outside can be seen (through a window or glass front); that
// second kind stays in the outdoor group but is marked "seen from inside". LanePulse places every image; the operator
// moves any image it got wrong (one at a time, or several picked together), can take a move back, and can teach
// LanePulse from those moves so the rest of this bundle, and the next ones, are sorted better. "Copy into folders"
// writes the groups to the exports folder; the source images are never changed.

type Label = "indoor" | "window" | "outdoor";   // window: outdoor, seen from inside
type Side = "indoor" | "outdoor";
type Item = { version_id: number; file: string; label: Label; group: Side; machine: Label; sure: number; decided: Label | null; decided_by: string | null; learned: boolean };
type Counts = { indoor: number; outdoor: number; window: number; moved: number; confirmed: number; unsure: number };
type Learner = { in_use: null | { version: number; created_at: string; decisions: number; indoor: number; window: number; outdoor: number; missions: number; right_before: number; right_after: number }; versions: number };
type Data = { items: Item[]; counts: Counts; learner: Learner; folder: string };

const SIDES: Side[] = ["indoor", "outdoor"];
const LABELS: Label[] = ["indoor", "outdoor", "window"];
const TITLE: Record<Side, string> = { indoor: "Indoor", outdoor: "Outdoor" };
const NAME: Record<Label, string> = { indoor: "indoor", outdoor: "outdoor, taken outside", window: "outdoor, seen from inside" };
// The two moves offered on a card, from where it is now.
const MOVES: Record<Label, [Label, string][]> = {
  indoor: [["outdoor", "Outdoor →"], ["window", "Seen from inside →"]],
  outdoor: [["indoor", "← Indoor"], ["window", "Mark seen from inside"]],
  window: [["indoor", "← Indoor"], ["outdoor", "Mark taken outside"]],
};
const PICK_MOVE: Record<Label, string> = { indoor: "Move to indoor", outdoor: "Move to outdoor, taken outside", window: "Move to outdoor, seen from inside" };
const PAGE = 48;
const UNSURE = 0.8;

export default function SortingTab({ m, reload }: { m: any; reload: () => void }) {
  const [data, setData] = useState<Data | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [order, setOrder] = useState<"unsure" | "name">("unsure");
  const [outShow, setOutShow] = useState<"all" | "outdoor" | "window">("all");
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [shown, setShown] = useState<Record<Side, number>>({ indoor: PAGE, outdoor: PAGE });
  const [note, setNote] = useState<{ kind: "ok" | "warn"; text: string; undo?: () => void } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [learnedAt, setLearnedAt] = useState<number | null>(null);   // decisions when the operator last pressed Learn
  const job = m.job;
  const running = job && ["running", "paused"].includes(job.state);

  const load = useCallback(() => api<Data>(`/missions/${m.id}/scenes`).then((d) => { setData(d); setErr(null); })
    .catch((e: ApiError) => setErr(e.message)), [m.id]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (running) { const t = setInterval(load, 2000); return () => clearInterval(t); } load(); return undefined; }, [running, load]);

  const groups = useMemo(() => {
    const g: Record<Side, Item[]> = { indoor: [], outdoor: [] };
    for (const it of data?.items ?? []) if (it.group === "indoor" || outShow === "all" || it.label === outShow) g[it.group].push(it);
    if (order === "unsure") for (const s of SIDES) g[s].sort((a, b) => (a.decided ? 1 : 0) - (b.decided ? 1 : 0) || (b.learned ? 1 : 0) - (a.learned ? 1 : 0) || a.sure - b.sure);
    return g;
  }, [data, order, outShow]);

  const decide = async (ids: number[], label: Label | null, quiet = false) => {
    if (!ids.length) return;
    setBusy("move");
    const before = new Map((data?.items ?? []).filter((i) => ids.includes(i.version_id)).map((i) => [i.version_id, i.decided]));
    try {
      await post(`/missions/${m.id}/scenes/decide`, { version_ids: ids, label });
      setPicked(new Set());
      await load();
      if (!quiet) {
        const what = ids.length === 1 ? (data?.items.find((i) => i.version_id === ids[0])?.file ?? "1 image") : `${ids.length} images`;
        setNote({ kind: "ok", text: label ? `Moved ${what} to ${NAME[label]}.` : `${what}: back to LanePulse's choice.`,
          undo: async () => {
            // Put each image back as it was: grouped by its earlier decision.
            const by = new Map<string, number[]>();
            before.forEach((d, id) => { const k = d ?? ""; by.set(k, [...(by.get(k) ?? []), id]); });
            for (const [k, v] of by) await post(`/missions/${m.id}/scenes/decide`, { version_ids: v, label: k || null });
            setNote(null); load();
          } });
      }
    } catch (e) { setNote({ kind: "warn", text: (e as ApiError).message }); }
    finally { setBusy(null); }
  };

  const learn = async () => {
    setBusy("learn"); setNote(null);
    try {
      const r = await post(`/missions/${m.id}/scenes/learn`);
      setLearnedAt((data?.counts.moved ?? 0) + (data?.counts.confirmed ?? 0));
      if (r.kept) {
        const before = data?.items ?? [];
        const now: Data = await api(`/missions/${m.id}/scenes`);
        setData(now);
        const changed = now.items.filter((i) => !i.decided && before.find((b) => b.version_id === i.version_id)?.machine !== i.machine).length;
        setNote({ kind: "ok", text: `Learned from ${r.decisions} image${r.decisions === 1 ? "" : "s"} you placed. ${changed ? `${changed} image${changed === 1 ? "" : "s"} you had not checked looked like them and moved; they are marked "like images you placed".` : "No image you had not checked looked enough like them to be moved."}` });
      } else {
        setNote({ kind: "warn", text: r.reason });
      }
    } catch (e) { setNote({ kind: "warn", text: (e as ApiError).message }); }
    finally { setBusy(null); }
  };

  const forget = async () => {
    setBusy("learn");
    try { await post(`/missions/${m.id}/scenes/forget`); await load(); setNote({ kind: "ok", text: "LanePulse is back to sorting the way it did before learning. What it learned is kept, and Learn brings it back." }); }
    catch (e) { setNote({ kind: "warn", text: (e as ApiError).message }); }
    finally { setBusy(null); }
  };

  const copy = async () => {
    setBusy("copy"); setNote(null);
    try {
      const r = await post(`/missions/${m.id}/scenes/copy`);
      setNote({ kind: r.missing ? "warn" : "ok", text: `Copied ${r.copied} images: ${r.indoor} into indoor and ${r.outdoor} into outdoor${r.window ? ` (${r.window} of them seen from inside, in outdoor/seen from inside)` : ""}, in ${r.folder}, with a list of them (sorting.csv).${r.missing ? ` ${r.missing} are no longer in the source folder.` : ""}` });
    } catch (e) { setNote({ kind: "warn", text: (e as ApiError).message }); }
    finally { setBusy(null); }
  };

  const toggle = (id: number) => setPicked((p) => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; });

  if (err) return <div className="msg err">{err}</div>;
  if (!data) return <p className="muted">Loading…</p>;
  const c = data.counts;
  const total = c.indoor + c.outdoor;
  const pickedItems = data.items.filter((i) => picked.has(i.version_id));
  const pickedSides = new Set(pickedItems.map((i) => i.label));
  const yours = c.moved + c.confirmed;
  const newSinceLearn = learnedAt === null ? c.moved : Math.max(0, yours - learnedAt);
  const pct = job && job.total ? Math.round((100 * job.done) / job.total) : 0;

  return (
    <div className="sorting">
      {running && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}><b>{job.state === "paused" ? "Paused" : "Sorting the images"}</b>
            <span className="muted">{job.done} of {job.total} · {job.eta_s != null ? `about ${fmtDuration(job.eta_s)} left` : ""}</span></div>
          <div className="progress" style={{ margin: "8px 0 0" }}><div style={{ width: `${pct}%` }} /></div>
        </div>
      )}
      {!running && total === 0 && (
        <div className="card">
          <h2>Nothing sorted yet</h2>
          <p className="muted">LanePulse looks at each image in the source folder and puts it in indoor or outdoor. File names do not need a date or time.</p>
          <button className="primary" onClick={async () => { await post(`/missions/${m.id}/job`).catch(() => undefined); reload(); }}>Sort the images</button>
        </div>
      )}

      {total > 0 && (
        <div className="sort-head">
          <div className="sort-figs">
            <div><b>{c.indoor}</b><span>Indoor</span></div>
            <div><b>{c.outdoor}</b><span>Outdoor</span></div>
            <div className="win"><b>{c.window}</b><span>of them seen from inside</span></div>
            <div><b>{c.moved}</b><span>Moved by you</span></div>
            <div className={c.unsure ? "warn" : ""}><b>{c.unsure}</b><span>LanePulse not sure</span></div>
          </div>
          <div className="sort-actions">
            <div className="seg compact" role="group" aria-label="Order">
              <button className={order === "unsure" ? "on" : ""} onClick={() => setOrder("unsure")} title="Images moved by what LanePulse learned come first, then the ones it is least sure of">Least sure first</button>
              <button className={order === "name" ? "on" : ""} onClick={() => setOrder("name")}>By name</button>
            </div>
            <button className={newSinceLearn > 0 ? "attn" : ""} disabled={!!busy || yours === 0} onClick={learn}
              title="Teach LanePulse from the images you moved or marked as right, then sort again the ones you have not checked">
              {busy === "learn" ? "Learning…" : newSinceLearn > 0 ? `Learn from your ${newSinceLearn} change${newSinceLearn === 1 ? "" : "s"}` : "Learn from your choices"}</button>
            <button className="primary" disabled={!!busy || running} onClick={copy}>{busy === "copy" ? "Copying…" : "Copy into indoor and outdoor folders"}</button>
          </div>
        </div>
      )}

      {note && <div className={`msg ${note.kind}`} role="status">{note.text}{note.undo && <> <button className="linkish" onClick={note.undo}>Undo</button></>}</div>}

      {picked.size > 0 && (
        <div className="sort-picked" role="toolbar" aria-label="Picked images">
          <b>{picked.size} picked</b>
          {LABELS.filter((s) => pickedItems.some((i) => i.label !== s)).map((s) =>
            <button key={s} className="primary" disabled={!!busy} onClick={() => decide(pickedItems.filter((i) => i.label !== s).map((i) => i.version_id), s)}>{PICK_MOVE[s]}</button>)}
          {pickedItems.some((i) => !i.decided) && pickedSides.size === 1 &&
            <button disabled={!!busy} onClick={() => decide(pickedItems.filter((i) => !i.decided).map((i) => i.version_id), [...pickedSides][0], true)}>Mark as right</button>}
          {pickedItems.some((i) => i.decided) && <button disabled={!!busy} onClick={() => decide(pickedItems.filter((i) => i.decided).map((i) => i.version_id), null)}>Back to LanePulse's choice</button>}
          <button className="linkish" onClick={() => setPicked(new Set())}>Clear</button>
        </div>
      )}

      {total > 0 && (
        <div className="sort-cols">
          {SIDES.map((s) => (
            <section key={s} className="card sort-col" aria-label={TITLE[s]}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2>{TITLE[s]} <span className="muted">{groups[s].length}</span></h2>
                {s === "outdoor" && c.window > 0 && (
                  <div className="seg compact" role="group" aria-label="Show">
                    <button className={outShow === "all" ? "on" : ""} onClick={() => setOutShow("all")}>All</button>
                    <button className={outShow === "outdoor" ? "on" : ""} onClick={() => setOutShow("outdoor")}>Taken outside</button>
                    <button className={outShow === "window" ? "on" : ""} onClick={() => setOutShow("window")}>Seen from inside</button>
                  </div>
                )}
                {groups[s].some((i) => !i.decided) &&
                  <button className="linkish" disabled={!!busy} title="Tell LanePulse every image shown here is in the right place; it learns from these too"
                    onClick={async () => {
                      // Each image is confirmed where it is: indoor, outdoor, or outdoor seen from inside.
                      const todo = groups[s].slice(0, shown[s]).filter((i) => !i.decided);
                      for (const l of LABELS) { const ids = todo.filter((i) => i.label === l).map((i) => i.version_id); if (ids.length) await decide(ids, l, true); }
                    }}>All shown are right</button>}
              </div>
              <div className="sort-grid">
                {groups[s].slice(0, shown[s]).map((it) => (
                  <figure key={it.version_id} className={`sort-card${picked.has(it.version_id) ? " on" : ""}${it.decided ? " yours" : ""}${it.label === "window" ? " window" : ""}${!it.decided && it.learned ? " learned" : !it.decided && it.sure < UNSURE ? " unsure" : ""}`}>
                    <button className="sort-pick" aria-pressed={picked.has(it.version_id)} onClick={() => toggle(it.version_id)} title="Pick, to move several together">
                      <img loading="lazy" src={`/api/missions/${m.id}/thumbs/${it.version_id}?w=320`} alt={it.file} />
                      <span className="tick" aria-hidden="true">{picked.has(it.version_id) ? "✓" : ""}</span>
                      {it.label === "window" && <span className="win-badge">Seen from inside</span>}
                    </button>
                    <figcaption>
                      <span className="name mono bidi" title={it.file}>{it.file.split(/[\\/]/).pop()}</span>
                      <span className="why">{it.decided
                        ? (it.decided !== it.machine ? <>Moved by you <button className="linkish" onClick={() => decide([it.version_id], null)}>undo</button></> : "You marked it right")
                        : it.learned ? "LanePulse, like images you placed" : `LanePulse, ${Math.round(it.sure * 100)}% sure`}</span>
                      <span className="acts">
                        <span className="moves">{MOVES[it.label].map(([to, text]) =>
                          <button key={to} disabled={!!busy} title={`Move to ${NAME[to]}`} onClick={() => decide([it.version_id], to)}>{text}</button>)}</span>
                        <a href={`/api/missions/${m.id}/images/${it.version_id}`} target="_blank" rel="noreferrer" title="Open the full image in a new tab" aria-label={`Open ${it.file} full size`}>{"\u2197"}</a>
                      </span>
                    </figcaption>
                  </figure>
                ))}
              </div>
              {groups[s].length > shown[s] && <button onClick={() => setShown((x) => ({ ...x, [s]: x[s] + PAGE * 2 }))}>Show {Math.min(PAGE * 2, groups[s].length - shown[s])} more</button>}
              {!groups[s].length && <p className="muted">No images here.</p>}
            </section>
          ))}
        </div>
      )}

      {total > 0 && (
        <div className="card sort-learn">
          <h2>What LanePulse has learned</h2>
          {data.learner.in_use ? (
            <>
              <p>LanePulse sorts with what it learned from {data.learner.in_use.decisions} images people placed ({data.learner.in_use.indoor} indoor, {data.learner.in_use.outdoor} outdoor taken outside, {data.learner.in_use.window} outdoor seen from inside) in {data.learner.in_use.missions} sorting mission{data.learner.in_use.missions === 1 ? "" : "s"} on this computer. Images that look like those follow them.</p>
              <button disabled={!!busy} onClick={forget}>Go back to sorting without what it learned</button>
            </>
          ) : (
            <p className="muted">Nothing learned yet: LanePulse sorts by comparing each image with descriptions of a room, of a room with the outside showing through windows, and of an outdoor scene. Move the images it got wrong, then press Learn: images that look like the ones you placed then follow your choice, in this mission and in every later sorting mission on this computer. It is used only if it gets your images right at least as often as before.</p>
          )}
          <p className="muted" style={{ marginBottom: 0 }}>Copies go to <span className="mono bidi">{data.folder}</span>. The images in the source folder are never changed.</p>
        </div>
      )}
    </div>
  );
}
