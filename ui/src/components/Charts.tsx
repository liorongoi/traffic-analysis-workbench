import { useEffect, useMemo, useRef, useState } from "react";

/** The drawing is as wide as its box in real pixels, so text and lines keep their size on any screen. */
function useWidth(fallback = 640): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(280, Math.round(el.clientWidth))));
    ro.observe(el);
    setW(Math.max(280, Math.round(el.clientWidth)));
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

// Small SVG charts for the learning page: the loss per epoch while a model trains, and one evaluation beside
// another. Blue and orange (validated for colour-blind separation on the dark surface), thin marks, a legend,
// and a tooltip on hover; the figures are also in the tables beside them.

export type EpochRow = { epoch: number; train_loss: number | null; val_loss: number | null };

function niceTicks(lo: number, hi: number, n = 4): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return out;
}

const fmt = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));

export function LossChart({ history, height = 220 }: { history: EpochRow[]; height?: number }) {
  const [show, setShow] = useState<"both" | "train" | "val">("both");
  const [hover, setHover] = useState<number | null>(null);
  const [wrap, W] = useWidth();
  const hasVal = history.some((h) => h.val_loss != null);
  const series = [
    { key: "train_loss" as const, name: "Training loss", color: "var(--series-1)", on: show !== "val" },
    { key: "val_loss" as const, name: "Validation loss", color: "var(--series-2)", on: show !== "train" && hasVal },
  ];
  const H = height, L = 48, R = 16, T = 12, B = 30;
  const vals = history.flatMap((h) => series.filter((s) => s.on).map((s) => h[s.key])).filter((v): v is number => v != null);
  const best = useMemo(() => {
    let b: EpochRow | null = null;
    for (const h of history) if (h.val_loss != null && (b == null || h.val_loss < (b.val_loss as number))) b = h;
    return b;
  }, [history]);
  if (!history.length) return null;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.08;
  lo = Math.max(0, lo - pad);
  hi += pad;
  const ticks = niceTicks(lo, hi);
  const e0 = history[0].epoch, e1 = history[history.length - 1].epoch;
  const x = (e: number) => L + (e1 === e0 ? (W - L - R) / 2 : ((e - e0) / (e1 - e0)) * (W - L - R));
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const xticks = niceTicks(e0, e1, Math.min(8, Math.max(1, e1 - e0))).filter((v) => Number.isInteger(v));
  const path = (k: "train_loss" | "val_loss") => history.filter((h) => h[k] != null)
    .map((h, i) => `${i ? "L" : "M"}${x(h.epoch).toFixed(1)},${y(h[k] as number).toFixed(1)}`).join("");
  const onMove = (ev: React.MouseEvent<SVGSVGElement>) => {
    const r = ev.currentTarget.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    let best = 0, d = Infinity;
    history.forEach((h, i) => { const dd = Math.abs(x(h.epoch) - px); if (dd < d) { d = dd; best = i; } });
    setHover(best);
  };
  const hv = hover != null ? history[hover] : null;
  return (
    <div className="viz" ref={wrap} style={{ position: "relative" }}>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
        <div className="legend" style={{ margin: 0 }}>
          {series.filter((s) => s.on).map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.name}</span>)}
          {best && show !== "train" && <span className="muted">◆ kept: epoch {best.epoch} (lowest validation loss)</span>}
        </div>
        <div className="seg compact" role="group" aria-label="Which loss to show">
          {([["both", "Both"], ["train", "Training"], ["val", "Validation"]] as const).map(([k, l]) => (
            <button key={k} className={show === k ? "on" : ""} aria-pressed={show === k} disabled={k === "val" && !hasVal}
              onClick={() => setShow(k)}>{l}</button>))}
        </div>
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Loss per epoch" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <g className="grid">{ticks.map((t) => <line key={t} x1={L} x2={W - R} y1={y(t)} y2={y(t)} />)}</g>
        <g className="axis">
          {ticks.map((t) => <text key={t} x={L - 6} y={y(t) + 4} textAnchor="end">{fmt(t)}</text>)}
          {xticks.map((t) => <text key={t} x={x(t)} y={H - B + 16} textAnchor="middle">{t}</text>)}
          <text x={W - R} y={H - 2} textAnchor="end">epoch</text>
        </g>
        {series.filter((s) => s.on).map((s) => (
          <g key={s.key}>
            <path d={path(s.key)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            {history.length <= 40 && history.filter((h) => h[s.key] != null).map((h) => (
              <circle key={h.epoch} cx={x(h.epoch)} cy={y(h[s.key] as number)} r={2.5} fill={s.color} stroke="var(--surface)" strokeWidth={1} />))}
          </g>))}
        {best && show !== "train" && (
          <rect x={x(best.epoch) - 5} y={y(best.val_loss as number) - 5} width={10} height={10} transform={`rotate(45 ${x(best.epoch)} ${y(best.val_loss as number)})`}
            fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />)}
        {hv && <line x1={x(hv.epoch)} x2={x(hv.epoch)} y1={T} y2={H - B} stroke="var(--line-strong)" strokeDasharray="3 3" />}
      </svg>
      {hv && (
        <div className="tip" style={{ left: `${(x(hv.epoch) / W) * 100}%`, top: 34, transform: x(hv.epoch) > W * 0.7 ? "translateX(-105%)" : "translateX(8px)" }}>
          <b>Epoch {hv.epoch}</b>
          {series.filter((s) => s.on).map((s) => <div key={s.key}><i style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: s.color, marginRight: 6 }} />
            {s.name}: {hv[s.key] == null ? "–" : (hv[s.key] as number).toFixed(4)}</div>)}
        </div>
      )}
    </div>
  );
}

export type CompareGroup = { label: string; prev: number | null | undefined; cur: number | null | undefined };

/** Two evaluations side by side: previous (orange) and current (blue) per measure, 0 to 100%. */
export function CompareBars({ groups, prevName, curName, height = 210 }: {
  groups: CompareGroup[]; prevName: string; curName: string; height?: number;
}) {
  const [hover, setHover] = useState<{ i: number; which: "prev" | "cur" } | null>(null);
  const [wrap, W] = useWidth();
  const H = height, L = 40, R = 8, T = 22, B = 28;
  const gw = (W - L - R) / Math.max(groups.length, 1);
  const bw = Math.min(34, gw * 0.3);
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const bar = (v: number | null | undefined, cx: number, color: string, i: number, which: "prev" | "cur") => {
    if (v == null) return <text className="axis" x={cx} y={H - B - 4} textAnchor="middle">–</text>;
    const top = y(v), h = Math.max(0, H - B - top);
    const r = Math.min(4, h / 2, bw / 2);
    // rounded data end, square baseline
    const d = `M${cx - bw / 2},${H - B} V${top + r} Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top} H${cx + bw / 2 - r} Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r} V${H - B} Z`;
    return <g onMouseEnter={() => setHover({ i, which })} onMouseLeave={() => setHover(null)}>
      <rect x={cx - bw / 2 - 4} y={T} width={bw + 8} height={H - T - B} fill="transparent" />
      <path d={d} fill={color} opacity={hover && (hover.i !== i || hover.which !== which) ? 0.55 : 1} />
    </g>;
  };
  const hg = hover ? groups[hover.i] : null;
  return (
    <div className="viz" ref={wrap} style={{ position: "relative" }}>
      <div className="legend">
        <span><i className="box" style={{ background: "var(--series-2)" }} />{prevName}</span>
        <span><i className="box" style={{ background: "var(--series-1)" }} />{curName}</span>
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${curName} compared with ${prevName}`}>
        <g className="grid">{ticks.map((t) => <line key={t} x1={L} x2={W - R} y1={y(t)} y2={y(t)} />)}</g>
        <g className="axis">{ticks.map((t) => <text key={t} x={L - 6} y={y(t) + 4} textAnchor="end">{t * 100}%</text>)}</g>
        {groups.map((g, i) => {
          const cx = L + gw * i + gw / 2;
          const d = g.prev != null && g.cur != null ? (g.cur - g.prev) * 100 : null;
          const topV = Math.max(g.prev ?? 0, g.cur ?? 0);
          return <g key={g.label}>
            {bar(g.prev, cx - bw / 2 - 1, "var(--series-2)", i, "prev")}
            {bar(g.cur, cx + bw / 2 + 1, "var(--series-1)", i, "cur")}
            {d != null && <text className="val" x={cx} y={y(topV) - 6} textAnchor="middle"
              style={{ fill: Math.abs(d) < 0.05 ? "var(--muted)" : d > 0 ? "var(--ok)" : "var(--bad)" }}>
              {Math.abs(d) < 0.05 ? "same" : `${d > 0 ? "▲ +" : "▼ "}${d.toFixed(1)} pts`}</text>}
            <text className="cat" x={cx} y={H - B + 17} textAnchor="middle">{g.label}</text>
          </g>;
        })}
      </svg>
      {hg && hover && (
        <div className="tip" style={{ left: `${((L + gw * hover.i + gw / 2) / W) * 100}%`, top: 30, transform: "translateX(-50%)" }}>
          <b>{hg.label}</b>
          <div>{prevName}: {hg.prev == null ? "–" : `${(hg.prev * 100).toFixed(1)}%`}</div>
          <div>{curName}: {hg.cur == null ? "–" : `${(hg.cur * 100).toFixed(1)}%`}</div>
        </div>
      )}
    </div>
  );
}
