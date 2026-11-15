import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, ApiError } from "../api";

// Choose folders on the computer LanePulse runs on. The field takes a typed or pasted path; Browse opens a folder
// window like a file explorer: places, drives and network on the left, the folder's subfolders as tiles, a clickable
// path on top. With `multiple`, every folder chosen is added to a list, so a mission can read several folders.

type Root = { name: string; path: string; group?: string };
type Listing = { path: string | null; parent: string | null; folders: { name: string; path: string }[]; more?: boolean;
  files: number; roots: Root[] };

// The places on the left, in the order a file explorer shows them.
const GROUPS: [string, string][] = [["places", "Quick access"], ["drives", "This computer"], ["network", "Network"]];
const trim = (p: string) => p.replace(/[\\/]+$/, "");
const same = (a: string | null | undefined, b: string) => !!a && trim(a).toLowerCase() === trim(b).toLowerCase();

// The path as clickable parts: C:\ › Users › Lior, \\server\share › a, / › home › lior.
function crumbs(path: string): { name: string; path: string }[] {
  const win = /^[a-z]:[\\/]/i.test(path) || path.startsWith("\\\\");
  const sep = win ? "\\" : "/";
  const unc = path.match(/^\\\\[^\\]+\\[^\\]+/);
  const head = unc ? unc[0] : win ? path.slice(0, 3) : "/";
  const rest = path.slice(head.length).split(/[\\/]+/).filter(Boolean);
  const out = [{ name: unc ? head : win ? head.slice(0, 2) : "/", path: unc ? head + "\\" : head }];
  let at = unc ? head : trim(head);
  for (const part of rest) { at = `${at}${sep}${part}`; out.push({ name: part, path: at }); }
  return out;
}

function PlaceIcon({ name, group }: { name: string; group?: string }) {
  const n = name.toLowerCase();
  const d = group === "network" ? "M4 5h16v9H4zM12 14v4M7 20h10"
    : group === "drives" ? "M3 14h18v5H3zM5 14l2-9h10l2 9M7 16.5h.01"
    : n === "home" ? "M3 11l9-7 9 7M5 10v10h14V10"
    : n === "desktop" ? "M3 4h18v12H3zM8 20h8M12 16v4"
    : n === "documents" ? "M6 3h8l4 4v14H6zM14 3v4h4M9 12h6M9 16h6"
    : n === "downloads" ? "M12 4v11M7 10l5 5 5-5M4 20h16"
    : n === "pictures" ? "M3 5h18v14H3zM3 16l5-5 4 4 3-3 6 6M15.5 9.5h.01"
    : n === "videos" || n === "movies" ? "M3 6h13v12H3zM16 10l5-3v10l-5-3"
    : "M3 6h7l2 2h9v11H3z";
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>;
}

const FolderIcon = () => (
  <svg viewBox="0 0 48 40" aria-hidden="true" className="fx-folder">
    <path d="M2 6a3 3 0 0 1 3-3h13l4 5h21a3 3 0 0 1 3 3v24a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3z" className="fx-folder-back" />
    <path d="M2 14a3 3 0 0 1 3-3h38a3 3 0 0 1 3 3v21a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3z" className="fx-folder-front" />
  </svg>
);

export default function FolderPicker({ value, onChange, multiple = false, placeholder, label }: {
  value: string[]; onChange: (v: string[]) => void; multiple?: boolean; placeholder?: string; label: string;
}) {
  const [text, setText] = useState(multiple ? "" : value[0] ?? "");
  const [open, setOpen] = useState(false);
  const [list, setList] = useState<Listing | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const grid = useRef<HTMLDivElement>(null);

  useEffect(() => { if (!multiple) setText(value[0] ?? ""); }, [value, multiple]);
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); setOpen(false); } };
    document.addEventListener("keydown", key, true);
    return () => document.removeEventListener("keydown", key, true);
  }, [open]);

  const browse = async (path: string | null, remember = true) => {
    setErr(null); setLoading(true); setSel(null); setFilter("");
    try {
      const r: Listing = await api(`/folders${path ? `?path=${encodeURIComponent(path)}` : ""}`);
      if (remember && list?.path && !same(list.path, r.path ?? "")) setHistory((h) => [...h.slice(-30), list.path!]);
      setList(r);
      grid.current?.scrollTo({ top: 0 });
    } catch (e) {
      setErr((e as ApiError).message);
      if (!list) setList(await api("/folders").catch(() => null));
    } finally { setLoading(false); }
  };
  const back = () => { const h = [...history]; const p = h.pop(); setHistory(h); if (p) browse(p, false); };
  const start = () => {
    setOpen(true); setHistory([]);
    // Several folders usually sit side by side: start next to the last one added.
    const last = multiple ? value[value.length - 1] : null;
    const beside = last ? last.replace(/[\\/]+$/, "").replace(/[\\/][^\\/]*$/, "") || last : null;
    browse(text.trim() || beside || null, false);
  };
  const choose = (path: string) => {
    if (!path) return;
    if (multiple) {
      if (!value.includes(path)) onChange([...value, path]);
      setText("");
    } else {
      onChange([path]);
      setText(path);
      setOpen(false);
    }
  };

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return (list?.folders ?? []).filter((x) => !f || x.name.toLowerCase().includes(f));
  }, [list, filter]);
  const target = sel ?? list?.path ?? null;
  const targetName = target ? (crumbs(target).slice(-1)[0]?.name ?? target) : "";

  const window_ = open && createPortal(
    <div className="fx-bg" onMouseDown={(e) => { if (e.target === e.currentTarget) setOpen(false); }}>
      <div className="fx" role="dialog" aria-modal="true" aria-label={label}>
        <header className="fx-head">
          <div className="fx-title"><span className="fx-kicker">Choose a folder</span><b>{label}</b></div>
          <button className="fx-x" onClick={() => setOpen(false)} aria-label="Close" title="Close (Esc)">×</button>
        </header>
        <div className="fx-bar">
          <button className="fx-nav" onClick={back} disabled={history.length === 0} aria-label="Back" title="Back">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" /></svg></button>
          <button className="fx-nav" onClick={() => list?.parent && browse(list.parent)} disabled={!list?.parent} aria-label="Up one level" title="Up one level">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7" /></svg></button>
          <nav className="fx-crumbs" aria-label="Path">
            {list?.path ? crumbs(list.path).map((c, i, a) => (
              <span key={c.path} className="fx-crumb">
                <button onClick={() => browse(c.path)} disabled={i === a.length - 1} className="bidi">{c.name}</button>
                {i < a.length - 1 && <span className="fx-sep" aria-hidden>›</span>}
              </span>)) : <span className="fx-crumb muted">This computer</span>}
          </nav>
          <input className="fx-filter" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter" aria-label="Filter the folders here"
            disabled={!list?.path} />
        </div>
        <div className="fx-body">
          <nav className="fx-side" aria-label="Places">
            {GROUPS.map(([g, title]) => {
              const items = (list?.roots ?? []).filter((r) => (r.group ?? "drives") === g);
              return items.length > 0 && <div key={g} className="fx-group">
                <h4>{title}</h4>
                {items.map((r) => <button key={r.path} className={`fx-place${same(list?.path, r.path) ? " on" : ""}`}
                  title={r.path} onClick={() => browse(r.path)}><PlaceIcon name={r.name} group={r.group} /><span className="bidi">{r.name}</span></button>)}
              </div>;
            })}
          </nav>
          <div className="fx-main" ref={grid}>
            {err && <div className="msg err" style={{ margin: "0 0 10px" }}>{err}</div>}
            {list && !list.path && (
              <div className="fx-tiles">
                {(list.roots ?? []).map((r) => (
                  <button key={r.path} className="fx-tile fx-drive" onClick={() => browse(r.path)} title={r.path}>
                    <span className="fx-ico"><PlaceIcon name={r.name} group={r.group} /></span><span className="fx-name bidi">{r.name}</span>
                  </button>))}
              </div>)}
            {list?.path && (
              <>
                {shown.length === 0 && !loading && <div className="fx-empty">{filter ? "No folder here matches the filter." : "No subfolders here."}</div>}
                <div className={`fx-tiles${loading ? " loading" : ""}`} role="listbox" aria-label="Folders">
                  {shown.map((f) => (
                    <div key={f.path} className={`fx-tile${same(sel, f.path) ? " on" : ""}${multiple && value.includes(f.path) ? " added" : ""}`}
                      role="option" aria-selected={same(sel, f.path)} tabIndex={0} title={`${f.name}: double-click to open`}
                      onClick={() => setSel(f.path)} onDoubleClick={() => browse(f.path)}
                      onKeyDown={(e) => { if (e.key === "Enter") browse(f.path); if (e.key === " ") { e.preventDefault(); setSel(f.path); } }}>
                      <span className="fx-ico"><FolderIcon /></span>
                      <span className="fx-name bidi">{f.name}</span>
                      {multiple && <button className="fx-add" onClick={(e) => { e.stopPropagation(); if (!value.includes(f.path)) onChange([...value, f.path]); }}
                        disabled={value.includes(f.path)} title={value.includes(f.path) ? "Added" : `Add ${f.name}`}>{value.includes(f.path) ? "✓" : "+"}</button>}
                    </div>))}
                </div>
                {list.more && <div className="fx-empty">Only the first 500 subfolders are shown; use the filter or type a path.</div>}
              </>)}
          </div>
        </div>
        <footer className="fx-foot">
          <div className="fx-status">
            {list?.path ? <>
              <span className="fx-count">{list.files ? `${list.files.toLocaleString()} image or video file${list.files === 1 ? "" : "s"}` : "No images or videos"} directly in this folder</span>
              {list.folders.length > 0 && <span className="muted"> · {list.folders.length}{list.more ? "+" : ""} folder{list.folders.length === 1 ? "" : "s"}</span>}
            </> : <span className="muted">Pick a place on the left. A network folder can be typed in the field, such as \\server\share.</span>}
            {multiple && value.length > 0 && <span className="fx-chosen">{value.length} folder{value.length === 1 ? "" : "s"} added</span>}
          </div>
          <div className="fx-actions">
            <button onClick={() => setOpen(false)}>{multiple ? "Done" : "Cancel"}</button>
            <button className="primary" disabled={!target || (multiple && value.includes(target))} onClick={() => target && choose(target)}
              title={target ?? ""}>
              <span>{multiple ? (target && value.includes(target) ? "Added" : "Add") : "Choose"}{target ? <> “<bdi>{targetName}</bdi>”</> : null}</span>
            </button>
          </div>
        </footer>
      </div>
    </div>, document.body);

  return (
    <div className="folderpick">
      <span className="folderpick-label">{label}</span>
      {multiple && value.length > 0 && (
        <ul className="folderlist" aria-label="Chosen folders">
          {value.map((f) => (
            <li key={f}>
              <span className="mono bidi">{f}</span>
              <button className="link" aria-label={`Remove ${f}`} title="Remove this folder" onClick={() => onChange(value.filter((x) => x !== f))}>×</button>
            </li>
          ))}
        </ul>
      )}
      <div className="folderpick-field">
        <input className="mono" value={text} placeholder={placeholder ?? "Type a path or press Browse"} aria-label={label}
          onChange={(e) => { setText(e.target.value); if (!multiple) onChange(e.target.value.trim() ? [e.target.value] : []); }}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); if (multiple && text.trim()) choose(text.trim()); else start(); }
            if (e.key === "ArrowDown") { e.preventDefault(); start(); }
          }} />
        {multiple && <button onClick={() => choose(text.trim())} disabled={!text.trim()}>Add</button>}
        <button aria-haspopup="dialog" aria-expanded={open} onClick={start} title="Browse the folders on this computer">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true"><path d="M3 6h7l2 2h9v11H3z" /></svg>
          Browse
        </button>
      </div>
      {window_}
    </div>
  );
}
