import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { go } from "../router";

// Jump to any mission, the images it still needs reviewed, or a page, from the app bar: Ctrl+K (⌘K on a Mac)
// focuses it, arrows pick, Enter goes. One shortcut instead of clicking through Missions, a mission and a tab.

type M = { id: string; name: string; input_type: string; review?: { unreviewed: number; first_unreviewed: number | null } | null };
type Item = { label: string; hint: string; to: string };

export default function JumpBar() {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [list, setList] = useState<M[]>([]);
  const [pick, setPick] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); input.current?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    if (!open) return;
    api("/missions").then((r) => setList(r.missions)).catch(() => setList([]));
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const items = useMemo(() => {
    const all: Item[] = [];
    for (const m of list) {
      const r = m.review;
      if (r && r.unreviewed > 0 && r.first_unreviewed) all.push({ label: `Review ${m.name}`, hint: `${r.unreviewed} to review`, to: `/m/${m.id}/i/${r.first_unreviewed}` });
      all.push({ label: m.name, hint: m.input_type === "video" ? "Video mission" : "Mission", to: `/m/${m.id}` });
      all.push({ label: `${m.name}: ${m.input_type === "video" ? "events" : "results"}`, hint: "Tab", to: `/m/${m.id}/${m.input_type === "video" ? "events" : "results"}` });
    }
    all.push({ label: "New mission", hint: "Page", to: "/new" }, { label: "All missions", hint: "Page", to: "/" },
      { label: "Filename formats", hint: "Settings", to: "/settings/formats" },
      { label: "Settings", hint: "Page", to: "/settings" }, { label: "Other computers that help", hint: "Settings", to: "/settings/workers" });
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return all.filter((it) => words.every((w) => it.label.toLowerCase().includes(w))).slice(0, 12);
  }, [list, q]);
  useEffect(() => setPick(0), [q]);

  const choose = (it: Item | undefined) => {
    if (!it) return;
    go(it.to);
    setOpen(false);
    setQ("");
    input.current?.blur();
  };
  const mac = typeof navigator !== "undefined" && /Mac/.test(navigator.platform);

  return (
    <div className="jump" ref={box}>
      <label className="jump-field">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
        <input ref={input} value={q} placeholder="Jump to a mission or page" aria-label="Jump to a mission or page"
          role="combobox" aria-expanded={open} aria-controls="jump-menu" aria-autocomplete="list"
          onFocus={() => setOpen(true)} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setPick((p) => Math.min(p + 1, items.length - 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setPick((p) => Math.max(p - 1, 0)); }
            else if (e.key === "Enter") { e.preventDefault(); choose(items[pick]); }
            else if (e.key === "Escape") { setOpen(false); input.current?.blur(); }
          }} />
        <kbd>{mac ? "⌘K" : "Ctrl K"}</kbd>
      </label>
      {open && (
        <div className="jump-menu" id="jump-menu" role="listbox">
          {items.length === 0 && <div className="none">Nothing matches “{q}”.</div>}
          {items.map((it, i) => (
            <button key={it.to + it.label} type="button" role="option" aria-selected={i === pick} className={i === pick ? "on" : ""}
              onMouseEnter={() => setPick(i)} onClick={() => choose(it)}>
              <span className="bidi">{it.label}</span><small>{it.hint}</small>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
