import { useEffect, useState } from "react";
import { ApiError } from "../api";

/** What a mission looks for: the vehicle classes and any of its own (a pedestrian, a tractor), with the words
 *  SAM 3.1 is asked for. Used in Settings (what new missions look for), on New mission and in a mission's setup. */
const CLASS_LABEL: Record<string, string> = { car: "Car", truck: "Truck", bus: "Bus", motorcycle: "Motorcycle", bicycle: "Bicycle" };

/** Road words offered in the drop-down beside each "add a word" box, grouped under the class they belong to. SAM 3.1
 *  has no fixed class list (it looks for whatever phrase it is given), so these are only suggestions. */
type Groups = [string, string[]][];
const SUGGESTED_WORDS: Record<string, Groups> = {
  car: [["Passenger cars", ["car", "taxi", "police car", "station wagon", "suv"]],
    ["Vans and pickups", ["van", "minivan", "pickup truck", "camper van", "motorhome"]],
    ["Cars with trailers", ["car with trailer", "car towing a caravan"]],
    ["Emergency", ["ambulance"]]],
  truck: [["Semi-trailers and full trailers", ["semi-trailer truck", "tractor-trailer", "articulated lorry", "truck with trailer", "car transporter"]],
    ["Rigid trucks", ["truck", "lorry", "box truck", "delivery truck"]],
    ["Special trucks", ["dump truck", "tanker truck", "fire truck", "garbage truck", "tow truck", "snow plough"]]],
  bus: [["Buses", ["bus", "minibus", "coach", "double-decker bus", "articulated bus", "school bus"]]],
  motorcycle: [["Motorcycles", ["motorcycle", "motorbike", "scooter", "moped"]]],
  bicycle: [["Bicycles", ["bicycle", "cyclist", "cargo bike", "electric scooter"]]],
};
/** Extra search words, so typing "hgv", "semi" or "camper" also finds a suggestion. */
const KEYWORDS: Record<string, string[]> = {
  "truck": ["hgv", "heavy"], "lorry": ["hgv", "heavy"], "semi-trailer truck": ["semi", "hgv", "articulated", "heavy"],
  "tractor-trailer": ["semi", "18-wheeler", "hgv", "heavy"], "articulated lorry": ["semi", "hgv", "heavy"],
  "truck with trailer": ["full trailer", "drawbar", "hgv", "heavy"], "car transporter": ["trailer", "hgv"],
  "tanker truck": ["hgv"], "car with trailer": ["trailer", "towing"], "car towing a caravan": ["trailer", "towing", "camper"],
  "camper van": ["camper", "rv"], "motorhome": ["camper", "rv"], "pickup truck": ["pickup"],
  "ambulance": ["emergency"], "police car": ["emergency"], "fire truck": ["emergency"],
  "coach": ["bus"], "minibus": ["bus"], "moped": ["two-wheeler"], "scooter": ["two-wheeler"], "motorcycle": ["two-wheeler", "mc"],
  "bicycle": ["bike", "two-wheeler"], "cyclist": ["bike"], "suv": ["jeep"], "taxi": ["cab"],
};
const matches = (w: string, q: string) => w.includes(q) || (KEYWORDS[w] ?? []).some((k) => k.includes(q));

/** Classes a mission may add, grouped, each with the words to start from. */
const SUGGESTED_CLASSES: [string, [string, string[]][]][] = [
  ["Other road users", [["pedestrian", ["pedestrian", "person"]]]],
  ["Farm and work vehicles", [["tractor", ["tractor"]], ["farm machinery", ["combine harvester", "tractor"]],
    ["construction vehicle", ["excavator", "wheel loader", "road roller"]], ["snow plough", ["snow plough", "snowplow"]]]],
  ["Emergency", [["emergency vehicle", ["ambulance", "fire truck", "police car"]]]],
  ["Trailers on their own", [["trailer", ["trailer", "caravan"]]]],
  ["Rail", [["tram", ["tram"]]]],
  ["Animals", [["animal", ["deer", "moose", "dog", "horse"]]]],
];
const CLASS_WORDS: Record<string, string[]> = Object.fromEntries(SUGGESTED_CLASSES.flatMap(([, cs]) => cs));

const classTitle = (c: string) => CLASS_LABEL[c] ?? c[0].toUpperCase() + c.slice(1);

/** A type-in box: type a leading word and every related suggestion is listed, grouped by kind (a group whose name
 *  matches is shown whole, so "trailer" lists all trailer combinations). Tick one or several and add them together,
 *  or press Enter to add the typed word itself. */
function AddPick({ groups, have, onAdd, disabled, label, prompt, maxLength, wide, targets }:
  { wide?: boolean; targets?: [string, string][]; groups: Groups; have: string[]; onAdd: (ws: string[], target?: string) => void; disabled: boolean; label: string; prompt: string; maxLength: number }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  const q = text.trim().toLowerCase();
  const shown: Groups = groups.map(([g, ws]): [string, string[]] => {
    const left = ws.filter((w) => !have.includes(w));
    return [g, !q || g.toLowerCase().includes(q) ? left : left.filter((w) => matches(w, q))];
  }).filter(([, ws]) => ws.length);
  const flat = shown.flatMap(([, ws]) => ws);
  const toggle = (w: string) => setPicked(picked.includes(w) ? picked.filter((x) => x !== w) : [...picked, w]);
  const toggleGroup = (ws: string[]) => {
    const all = ws.every((w) => picked.includes(w));
    setPicked(all ? picked.filter((w) => !ws.includes(w)) : [...picked, ...ws.filter((w) => !picked.includes(w))]);
  };
  const commit = (ws: string[], target?: string) => {
    const clean = ws.map((w) => w.trim()).filter(Boolean);
    if (clean.length) onAdd(clean, target);
    setText(""); setPicked([]); setOpen(false);
  };
  return (
    <div className={wide ? "addpick wide" : "addpick"} onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setOpen(false); }}>
      <input aria-label={label} placeholder={prompt} value={text} maxLength={maxLength} disabled={disabled}
        role="combobox" aria-expanded={open} autoComplete="off"
        onFocus={() => setOpen(true)} onChange={(e) => { setText(e.target.value); setOpen(true); }}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); if (picked.length) commit(picked); else if (!targets) commit([text]); }
          else if (e.key === "Escape") { setText(""); setPicked([]); setOpen(false); }
        }} />
      {open && !disabled && (shown.length > 0 || q) && (
        <div className="addpick-menu" role="listbox" aria-multiselectable="true" tabIndex={-1}>
          {q && !flat.includes(q) && !picked.length && !targets && (
            <button type="button" className="addpick-own" onMouseDown={(e) => e.preventDefault()} onClick={() => commit([text])}>Add “{text.trim()}”</button>
          )}
          {q && !flat.includes(q) && !picked.length && targets && (
            <div className="addpick-targets">Add “{text.trim()}” to
              {targets.map(([t, l]) => <button type="button" key={t} onMouseDown={(e) => e.preventDefault()} onClick={() => commit([text], t)}>{l}</button>)}
            </div>
          )}
          <div className="addpick-list">
            {shown.map(([g, ws]) => (
              <div key={g}>
                <label className="addpick-group" onMouseDown={(e) => e.preventDefault()}>
                  <input type="checkbox" checked={ws.every((w) => picked.includes(w))} onChange={() => toggleGroup(ws)} /> {g}
                </label>
                {ws.map((w) => (
                  <label key={w} className="addpick-opt" role="option" aria-selected={picked.includes(w)} onMouseDown={(e) => e.preventDefault()}>
                    <input type="checkbox" checked={picked.includes(w)} onChange={() => toggle(w)} /> {w}
                  </label>
                ))}
              </div>
            ))}
          </div>
          {picked.length > 0 && (
            <div className="addpick-foot">
              <button type="button" className="linkish" onMouseDown={(e) => e.preventDefault()} onClick={() => setPicked([])}>Clear</button>
              <button type="button" className="primary" onMouseDown={(e) => e.preventDefault()} onClick={() => commit(picked)}>Add {picked.length}</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** The words SAM 3.1 is asked for, per class: each word found counts as that class. Classes of the mission's own
 *  (e.g. pedestrian, tractor) can be added, and left out of the vehicle count. Folded to one summary line until opened. */
/** Which detector of the mission finds a class: the standard one knows the vehicle classes and some others (a
 *  person); anything else only SAM 3.1 can find. */
function foundByText(c: string, mine: boolean, knows: string[], samOnly?: boolean, standardOnly?: boolean): string {
  const std = !mine || knows.includes(c);
  if (samOnly) return "SAM 3.1";
  if (standardOnly) return std ? "standard" : "needs SAM 3.1 or Both";
  return std ? "standard + SAM 3.1" : "SAM 3.1 only";
}

export function TargetsEditor({ words: savedWords, uncounted: savedUncounted, defaults, busy, onSave, samOnly, standardOnly, standardKnows = [], title, note }: {
  words: Record<string, string[]> | null; uncounted: string[] | null; defaults: Record<string, string[]>; busy: boolean;
  onSave: (words: Record<string, string[]> | null, uncounted: string[]) => Promise<void>;
  samOnly?: boolean; standardOnly?: boolean; standardKnows?: string[]; title?: string; note?: string;
}) {
  // A vehicle class missing from the saved words keeps its default words, as on the server.
  const saved: Record<string, string[]> = savedWords ? { ...defaults, ...savedWords } : defaults;
  const savedOut: string[] = savedUncounted ?? [];
  const [open, setOpen] = useState(false);
  const [words, setWords] = useState<Record<string, string[]>>(saved);
  const [out, setOut] = useState<string[]>(savedOut);
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => { setWords(savedWords ? { ...defaults, ...savedWords } : defaults); setOut(savedUncounted ?? []); }, [JSON.stringify(savedWords), JSON.stringify(savedUncounted)]);
  const builtIn = Object.keys(defaults);
  const own = Object.keys(words).filter((c) => !builtIn.includes(c));
  const classes = [...builtIn, ...own];
  const used = Object.values(words).flat();
  const key = (w: Record<string, string[]>, o: string[]) => JSON.stringify([Object.keys(w).sort().map((c) => [c, w[c]]), [...o].sort()]);
  const changed = key(words, out) !== key(saved, savedOut);
  const isDefault = key(words, out) === key(defaults, []);
  const norm = (t: string) => t.trim().toLowerCase().replace(/\s+/g, " ");
  const remove = (c: string, w: string) => setWords({ ...words, [c]: (words[c] ?? []).filter((x) => x !== w) });
  const addClass = (ts: string[]) => {
    setErr(null);
    const names = ts.map(norm).filter((n, i, a) => n && a.indexOf(n) === i);
    const dup = names.filter((n) => classes.includes(n));
    if (dup.length) { setErr(`There is already a class called ${dup.join(", ")}.`); return; }
    const next = { ...words };
    for (const n of names) next[n] = CLASS_WORDS[n] ?? [n];
    setWords(next);
    setOut([...out, ...names]);  // a new class is not a vehicle until said so
  };
  const dropClass = (c: string) => {
    const { [c]: _, ...rest } = words;
    setWords(rest);
    setOut(out.filter((x) => x !== c));
  };
  const save = async (body: Record<string, string[]> | null, uncounted: string[]) => {
    setErr(null);
    setSaving(true);
    try { await onSave(body, uncounted); setOpen(false); }
    catch (e) { setErr((e as ApiError).message); }
    finally { setSaving(false); }
  };
  // One search for all classes: each suggestion is listed under "Class · group" and lands in that class.
  const searchGroups: Groups = classes.flatMap((c): Groups => (SUGGESTED_WORDS[c] ?? (CLASS_WORDS[c] ? [["Suggested", CLASS_WORDS[c]]] : []))
    .map(([g, ws]) => [`${classTitle(c)} · ${g}`, ws]));
  const wordClass: Record<string, string> = Object.fromEntries(searchGroups.flatMap(([g, ws]) =>
    ws.map((w) => [w, classes.find((c) => g.startsWith(classTitle(c) + " · "))!])));
  const addFound = (ws: string[], target?: string) => {
    const next = { ...words };
    for (const t of ws) {
      const w = norm(t), c = target ?? wordClass[w];
      if (!w || !c || (next[c] ?? []).includes(w)) continue;
      next[c] = [...(next[c] ?? []), w];
    }
    setWords(next);
  };
  const classGroups: Groups = SUGGESTED_CLASSES.map(([g, cs]) => [g, cs.map(([c]) => c)]);
  return (
    <div className="samwords">
      <div className="row samwords-head">
        <h3>{title ?? (standardOnly ? "What this mission looks for" : "What this mission looks for, and the words SAM 3.1 uses")}</h3>
        <button className="linkish" onClick={() => { if (open && changed) { setWords(saved); setOut(savedOut); } setOpen(!open); }}>
          {open ? (changed ? "Cancel" : "Close") : "Edit"}</button>
      </div>
      {!open && (
        <p className="samwords-sum">
          {classes.filter((c) => (saved[c] ?? []).length).map((c, i) => (
            <span key={c}>{i > 0 && " · "}<b>{classTitle(c)}</b>{out.includes(c) ? " (not counted)" : ""}: {(saved[c] ?? []).join(", ")}</span>
          ))}
        </p>
      )}
      {open && <>
        <div className="samwords-search">
          <AddPick wide groups={searchGroups} have={used} disabled={busy} label="Find words to add"
            prompt="Choose words to add ▾ (typing narrows the list)" maxLength={40}
            targets={classes.map((c) => [c, classTitle(c)])} onAdd={addFound} />
        </div>
        <table className="samwords-table">
          <thead><tr><th>Class</th><th>Search words</th><th>Found by</th><th title="Counted as a vehicle">Counted</th><th /></tr></thead>
          <tbody>
            {classes.map((c) => {
              const mine = own.includes(c);
              return (
                <tr key={c}>
                  <td className="cls">{classTitle(c)}</td>
                  <td>
                    <div className="chips">
                      {(words[c] ?? []).map((w) => (
                        <span key={w} className="wordchip">{w}
                          <button aria-label={`Remove ${w}`} disabled={busy} onClick={() => remove(c, w)}>×</button>
                        </span>
                      ))}
                      {(words[c] ?? []).length === 0 && <span className="muted">not looked for</span>}
                    </div>
                  </td>
                  <td className="small">{foundByText(c, mine, standardKnows, samOnly, standardOnly)}</td>
                  <td className="ctr">
                    <input type="checkbox" aria-label={`Count ${c} as a vehicle`} checked={!out.includes(c)} disabled={busy || !mine}
                      title={mine ? "Count what this class finds as vehicles" : "Vehicle classes are always counted"}
                      onChange={(e) => setOut(e.target.checked ? out.filter((x) => x !== c) : [...out, c])} />
                  </td>
                  <td className="ctr">
                    {mine && <button className="iconbtn" aria-label={`Remove class ${c}`} title="Remove class" disabled={busy} onClick={() => dropClass(c)}>×</button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="row samwords-foot">
          <AddPick wide groups={classGroups.map(([g, cs]) => [g, cs.filter((c) => !classes.includes(c))])} have={[]} disabled={busy}
            label="Add a class" prompt="+ Choose a class ▾" maxLength={24} onAdd={addClass} />
          <span className="spacer" />
          {!isDefault && <button className="linkish" disabled={busy || saving} onClick={() => save(null, [])}>Reset to defaults</button>}
          {changed && <button className="primary" disabled={busy || saving} onClick={() => save(words, out)}>{saving ? "Saving…" : "Save"}</button>}
        </div>
        <p className="muted small">SAM 3.1 has no fixed class list: choose suggested words from the list (typing only narrows it, or adds a word of your own). Each word adds about half a second per image. Classes you add are reviewed and exported like the others, and are not used to train the standard detector.</p>
      </>}
      {note && !open && <p className="muted small" style={{ margin: "4px 0 0" }}>{note}</p>}
      {err && <div className="msg err" role="alert">{err}</div>}
    </div>
  );
}
