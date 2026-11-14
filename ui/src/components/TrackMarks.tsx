// What the player draws for each target besides its box: the tail of where it has been (fading with age, with a
// ring where it stood still) and a small label with its ID, coloured by how sure the detector was. Kept small and
// thinned out in a busy scene: a label that would cover another one is left out (its box still shows, and pointing
// at the box tells all of it).

export type TailLen = "off" | "1" | "3" | "10" | "all";
export const TAILS: Record<TailLen, string> = { off: "No tail", "1": "Tail 1 s", "3": "Tail 3 s", "10": "Tail 10 s", all: "Whole track" };

type Ev = { id: number; key: string; cls: string; status: string; boxes: number[][]; scores?: (number | null)[] };
type Look = { stroke: string; dash: string };

const STILL = 0.25;          // box heights a second: slower than this is standing still
const STOP_S = 1.0;          // standing this long or more is a stop worth marking

/** The detector's score on the analysed frame nearest t (null: a box the operator drew, or none near). */
export function scoreAt(e: Ev, t: number): number | null {
  const b = e.boxes, s = e.scores;
  if (!s || !b.length) return null;
  let lo = 0, hi = b.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (b[mid][0] <= t) lo = mid; else hi = mid; }
  const i = Math.abs(b[lo][0] - t) <= Math.abs(b[hi][0] - t) ? lo : hi;
  return Math.abs(b[i][0] - t) <= 0.6 ? s[i] ?? null : null;
}

/** Red (unsure) through amber to green (sure): 0.2 or less is red, 0.8 or more green. */
export function scoreColour(s: number | null): string {
  if (s == null) return "#e6edf3";
  const k = Math.max(0, Math.min(1, (s - 0.2) / 0.6));
  return `hsl(${Math.round(k * 125)}, 85%, ${Math.round(60 - 6 * k)}%)`;
}

/** "E-0012" -> "E-12": short, and still the ID the events list shows. */
export const shortKey = (k: string) => k.replace(/-0+(\d)/, "-$1");

const foot = (b: number[]) => [(b[1] + b[3]) / 2, b[4]];        // [t, x1, y1, x2, y2] -> where it meets the road

/** Each point averaged with its two neighbours on each side: a box jitters by a few pixels from frame to frame, and
 *  a waiting target's tail would otherwise be a scribble. */
function smooth(p: number[][]): number[][] {
  return p.map((_, i) => {
    const w = p.slice(Math.max(0, i - 2), i + 3);
    return [w.reduce((a, v) => a + v[0], 0) / w.length, w.reduce((a, v) => a + v[1], 0) / w.length];
  });
}

/** Stretches where the target stood still for STOP_S or longer: where, from, to. */
function stops(b: number[][]): { x: number; y: number; t0: number; t1: number }[] {
  const out: { x: number; y: number; t0: number; t1: number }[] = [];
  let i = 0;
  while (i < b.length - 1) {
    let j = i;
    const h = Math.max(b[i][4] - b[i][2], 1);
    const [x0, y0] = foot(b[i]);
    // grow the stretch while the target stays within reach of where it stopped
    while (j + 1 < b.length) {
      const [x, y] = foot(b[j + 1]);
      const dt = b[j + 1][0] - b[i][0];
      if (Math.hypot(x - x0, y - y0) > Math.max(STILL * h * Math.max(dt, 1), 0.15 * h)) break;
      j++;
    }
    if (b[j][0] - b[i][0] >= STOP_S) {
      const [x, y] = foot(b[Math.floor((i + j) / 2)]);
      out.push({ x, y, t0: b[i][0], t1: b[j][0] });
      i = j + 1;
    } else i++;
  }
  return out;
}

/** Tails, as dots fading with age: drawn under the boxes. ``px`` is video pixels per screen pixel. */
export function Tails({ events, now, len, sel, px, lookOf, boxAt }: {
  events: Ev[]; now: number; len: TailLen; sel: number | null; px: number;
  lookOf: (e: any) => Look; boxAt: (e: any, t: number) => number[] | null;
}) {
  if (len === "off") return null;
  const span = len === "all" ? Infinity : Number(len);
  return <g className="tails" pointerEvents="none">
    {events.map((e) => {
      const cur = boxAt(e, now);
      if (!cur) return null;
      const past = e.boxes.filter((b) => b[0] < now - 0.02 && b[0] >= now - span);
      if (!past.length) return null;
      const pts = smooth(past.map(foot));
      const ts = past.map((b) => b[0]);
      const on = e.id === sel, col = lookOf(e).stroke;
      // one dot per analysed frame, smaller and fainter the older it is
      const age = Math.min(span, now - ts[0]) || 1;
      const st = stops(past).filter((s) => s.t1 >= now - span);
      return <g key={e.id} fill={col}>
        {pts.map(([x, y], i) => {
          const k = Math.max(0, 1 - (now - ts[i]) / age);          // 1 newest, 0 oldest
          return <circle key={i} cx={x} cy={y} r={((on ? 1.4 : 1) + (on ? 1.6 : 1.3) * k) * px}
            fillOpacity={(on ? 0.3 : 0.15) + (on ? 0.65 : 0.65) * k} />;
        })}
        {st.map((s, i) => <g key={i}>
          <circle cx={s.x} cy={s.y} r={(on ? 4 : 3) * px} fill="none" stroke={col} strokeWidth={1.2 * px} strokeOpacity={on ? 0.95 : 0.6} />
          {on && <text x={s.x + 6 * px} y={s.y + 3.5 * px} fontSize={10 * px} fill={col} paintOrder="stroke" stroke="#000"
            strokeWidth={2.5 * px}>stood {(s.t1 - s.t0).toFixed(1)} s</text>}
        </g>)}
      </g>;
    })}
  </g>;
}

/** Labels: drawn over everything. A label that would overlap one already placed is left out; the picked target's
 *  goes first and says more. */
export function Labels({ events, now, sel, px, showId, showScore, boxAt, statusText, W }: {
  events: Ev[]; now: number; sel: number | null; px: number; showId: boolean; showScore: boolean;
  boxAt: (e: any, t: number) => number[] | null; statusText: (e: any) => string; W: number;
}) {
  if (!showId && !showScore) return null;
  const fs = 10 * px, ch = 6.1 * px;                 // font size and the width of one character, in video pixels
  const items = events.map((e) => {
    const b = boxAt(e, now);
    if (!b) return null;
    const s = scoreAt(e, now);
    const on = e.id === sel;
    const parts = [showId || on ? shortKey(e.key) : "", showScore && s != null ? s.toFixed(2) : "",
      on ? e.cls : "", on ? statusText(e) : ""].filter(Boolean);
    return { e, b, s, on, text: parts.join(" ") };
  }).filter(Boolean) as { e: Ev; b: number[]; s: number | null; on: boolean; text: string }[];
  // the picked one first, then the surer and bigger ones
  items.sort((a, b) => Number(b.on) - Number(a.on) || (b.s ?? 1) - (a.s ?? 1) || (b.b[3] - b.b[1]) - (a.b[3] - a.b[1]));
  const placed: number[][] = [];
  const out = [];
  for (const it of items) {
    const w = it.text.length * ch + 4 * px, h = fs + 2 * px;
    const x = Math.min(Math.max(0, it.b[0]), W - w);
    let y = it.b[1] - 2 * px;                       // baseline just above the box; inside it at the top edge
    if (y - fs < 0) y = it.b[1] + fs + 1 * px;
    const r = [x, y - fs, x + w, y - fs + h];
    if (!it.on && placed.some((p) => r[0] < p[2] && r[2] > p[0] && r[1] < p[3] && r[3] > p[1])) continue;
    placed.push(r);
    out.push(<text key={it.e.id} x={x + 1 * px} y={y} fontSize={fs} fill={scoreColour(it.s)} paintOrder="stroke" stroke="#000"
      strokeWidth={2.5 * px} strokeOpacity={0.85} fontWeight={it.on ? 700 : 500} className="tlabel">{it.text}</text>);
  }
  return <g className="labels" pointerEvents="none">{out}</g>;
}

/** The colour key, beside the checkboxes. */
export function ScoreKey() {
  return <span className="score-key" title="Label colour: how sure the detector was of the target on this frame">
    <span>unsure</span><i style={{ background: `linear-gradient(90deg, ${[0.2, 0.4, 0.6, 0.8].map(scoreColour).join(", ")})` }} /><span>sure</span>
  </span>;
}
