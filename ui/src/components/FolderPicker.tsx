import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";

// Choose folders on the computer LanePulse runs on: click the field (or Browse) and walk the folders; typing a path is
// only a shortcut. With
// `multiple`, every folder chosen is added to a list, so a mission can read several folders.

type Listing = { path: string | null; parent: string | null; folders: { name: string; path: string }[]; more?: boolean;
  files: number; roots: { name: string; path: string }[] };

export default function FolderPicker({ value, onChange, multiple = false, placeholder, label }: {
  value: string[]; onChange: (v: string[]) => void; multiple?: boolean; placeholder?: string; label: string;
}) {
  const [text, setText] = useState(multiple ? "" : value[0] ?? "");
  const [open, setOpen] = useState(false);
  const [list, setList] = useState<Listing | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => { if (!multiple) setText(value[0] ?? ""); }, [value, multiple]);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const browse = async (path: string | null) => {
    setErr(null);
    try {
      setList(await api(`/folders${path ? `?path=${encodeURIComponent(path)}` : ""}`));
    } catch (e) {
      setErr((e as ApiError).message);
      if (!list) setList(await api("/folders").catch(() => null));
    }
  };
  const toggle = () => {
    if (open) { setOpen(false); return; }
    setOpen(true);
    // Several folders usually sit side by side: start next to the last one added.
    const last = multiple ? value[value.length - 1] : null;
    const beside = last ? last.replace(/[\\/]+$/, "").replace(/[\\/][^\\/]*$/, "") || last : null;
    browse(text.trim() || beside || null);
  };
  const choose = (path: string) => {
    if (!path) return;
    if (multiple) {
      if (!value.includes(path)) onChange([...value, path]);
      setText("");
    } else {
      onChange([path]);
      setText(path);
    }
    setOpen(false);
  };

  return (
    <div className="folderpick" ref={box}>
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
        <input className="mono" value={text} placeholder={placeholder ?? "Click to choose a folder"} aria-label={label}
          onClick={() => { if (!open) toggle(); }}
          onChange={(e) => { setText(e.target.value); if (!multiple) onChange(e.target.value.trim() ? [e.target.value] : []); }}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); if (multiple) choose(text.trim()); else { setOpen(true); browse(text.trim()); } }
            if (e.key === "ArrowDown") { e.preventDefault(); if (!open) toggle(); }
            if (e.key === "Escape") setOpen(false);
          }} />
        {multiple && <button onClick={() => choose(text.trim())} disabled={!text.trim()}>Add</button>}
        <button aria-haspopup="listbox" aria-expanded={open} onClick={toggle} title="Browse the folders on this computer">
          Browse <span aria-hidden>▾</span>
        </button>
      </div>
      {open && (
        <div className="folderpick-menu" role="listbox" aria-label="Folders">
          <div className="folderpick-roots">
            {list?.roots.map((r) => <button key={r.path} className="chip" onClick={() => browse(r.path)}>{r.name}</button>)}
          </div>
          {list?.path && <div className="folderpick-here">
            <span className="mono bidi" title={list.path}>{list.path}</span>
            <span className="muted">{list.files ? `${list.files} image or video file${list.files === 1 ? "" : "s"} here` : "no images directly here"}</span>
          </div>}
          {err && <div className="msg err" style={{ margin: "6px 0" }}>{err}</div>}
          <div className="folderpick-items">
            {list?.parent && <button className="folderpick-item" onClick={() => browse(list.parent)}>
              <span aria-hidden className="ico">↰</span> Up one level</button>}
            {list?.folders.map((f) => (
              <div key={f.path} className="folderpick-row">
                <button className="folderpick-item" role="option" aria-selected={value.includes(f.path)} onClick={() => browse(f.path)}
                  title={`Open ${f.name}`}><span aria-hidden className="ico">▸</span><span className="bidi">{f.name}</span></button>
                {multiple && <button className="folderpick-add" onClick={() => { if (!value.includes(f.path)) onChange([...value, f.path]); }} disabled={value.includes(f.path)}>
                  {value.includes(f.path) ? "Added" : "Add"}</button>}
              </div>
            ))}
            {list?.path && list.folders.length === 0 && <div className="muted" style={{ padding: "6px 8px" }}>No subfolders.</div>}
            {list?.more && <div className="muted" style={{ padding: "6px 8px" }}>Only the first 500 subfolders are listed; type a path to go further.</div>}
          </div>
          {list?.path && <div className="row" style={{ justifyContent: "flex-end", marginTop: 8 }}>
            <button className="primary" disabled={multiple && value.includes(list.path)} onClick={() => choose(list.path!)}>
              {multiple ? (value.includes(list.path) ? "This folder is added" : "Add this folder") : "Choose this folder"}</button>
          </div>}
        </div>
      )}
    </div>
  );
}
