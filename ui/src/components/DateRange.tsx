import { useEffect, useMemo, useRef, useState } from "react";

/** A date range in the app's own look, written the way the mission's country writes dates (day first everywhere but
 *  the United States), instead of the browser's picker, which follows the Windows language (mm/dd/yyyy on English
 *  Windows). Values are local dates as YYYY-MM-DD, or "" for no limit. Nielsen #2: match the real world. */
export default function DateRange({ from, to, onChange, country, near }: {
  from: string; to: string; onChange: (from: string, to: string) => void; country?: string | null;
  near?: string | null;   // a date in the data, so the calendar opens where the images are, not on today
}) {
  const [open, setOpen] = useState(false);
  const [pick, setPick] = useState<string | null>(null);   // the first day clicked, waiting for the second
  const start = from || near || iso(new Date());
  const [month, setMonth] = useState(() => start.slice(0, 7));
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => { if (open) { setMonth((from || near || iso(new Date())).slice(0, 7)); setPick(null); } }, [open]);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, [open]);

  const us = (country ?? "").toUpperCase() === "US";
  const show = (d: string) => { const [y, m, dd] = d.split("-"); return us ? `${m}/${dd}/${y}` : `${dd}/${m}/${y}`; };
  const label = !from && !to ? "Any date" : from === to ? show(from) : `${from ? show(from) : "…"} – ${to ? show(to) : "…"}`;

  const days = useMemo(() => {
    const [y, m] = month.split("-").map(Number);
    const first = new Date(Date.UTC(y, m - 1, 1));
    const lead = (first.getUTCDay() + 6) % 7;            // weeks start on Monday
    const n = new Date(Date.UTC(y, m, 0)).getUTCDate();
    return [...Array(lead).fill(null), ...Array.from({ length: n }, (_, i) => `${month}-${String(i + 1).padStart(2, "0")}`)];
  }, [month]);
  const shift = (k: number) => {
    const [y, m] = month.split("-").map(Number);
    const t = y * 12 + (m - 1) + k;
    setMonth(`${Math.floor(t / 12)}-${String((t % 12) + 1).padStart(2, "0")}`);
  };
  const click = (d: string) => {
    if (!pick) { setPick(d); onChange(d, d); return; }
    const [a, b] = pick <= d ? [pick, d] : [d, pick];
    onChange(a, b); setPick(null); setOpen(false);
  };
  const today = iso(new Date());
  const ago = (n: number) => iso(new Date(Date.now() - n * 86400000));
  const presets: [string, string, string][] = [["Any date", "", ""], ["Today", today, today], ["Yesterday", ago(1), ago(1)], ["Last 7 days", ago(6), today]];
  const title = new Date(`${month}-01T00:00:00Z`).toLocaleDateString(undefined, { month: "long", year: "numeric", timeZone: "UTC" });

  return (
    <div className="daterange" ref={box}>
      <button type="button" className={`daterange-btn${from || to ? " set" : ""}`} aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(!open)}>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="3.5" y="5" width="17" height="15" rx="2.5" /><path d="M3.5 10h17M8 3v4M16 3v4" /></svg>
        {label}
      </button>
      {open && <div className="daterange-pop" role="dialog" aria-label="Choose dates">
        <div className="daterange-presets">{presets.map(([l, a, b]) =>
          <button key={l} type="button" className={from === a && to === b ? "on" : ""} onClick={() => { onChange(a, b); setOpen(false); }}>{l}</button>)}</div>
        <div className="daterange-cal">
          <div className="daterange-head">
            <button type="button" aria-label="Previous month" onClick={() => shift(-1)}>‹</button>
            <b>{title}</b>
            <button type="button" aria-label="Next month" onClick={() => shift(1)}>›</button>
          </div>
          <div className="daterange-grid">
            {["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"].map((w) => <span key={w} className="wd">{w}</span>)}
            {days.map((d, i) => d === null ? <span key={`e${i}`} /> : (
              <button key={d} type="button" onClick={() => click(d)} aria-label={show(d)}
                className={[d === today ? "today" : "", from && to && d >= from && d <= to ? "in" : "", d === from || d === to ? "end" : ""].join(" ")}>
                {Number(d.slice(8))}</button>))}
          </div>
          <p className="daterange-hint">{pick ? "Now click the last day." : "Click the first day, then the last."}</p>
        </div>
      </div>}
    </div>
  );
}

function iso(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
