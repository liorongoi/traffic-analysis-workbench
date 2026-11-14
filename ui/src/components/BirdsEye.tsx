import { useEffect, useMemo, useRef, useState } from "react";

// The road from above: the playing video laid flat on the ground through the camera's ground homography (the frame
// is warped on the graphics card, every frame), and each vehicle drawn as its footprint on the road, with its track
// so far and its speed. Needs the camera on the map (points picked on the frame and the map) or flight data.

export type GroundSample = { t: number; G: number[]; top_row: number; near: number[][]; far: number[][] };
export type GroundView = {
  origin: [number, number]; frame_w: number; frame_h: number; how: string; registered: boolean; fixed: boolean;
  step_s: number | null; samples: GroundSample[];
};
type Ev = { id: number; key: string; cls: string; first_t: number; last_t: number; boxes: number[][] };
type Look = { stroke: string; dash: string };

// A typical footprint (length, width in metres) for each class: the box on the frame gives where, not how big.
const SIZE: Record<string, [number, number]> = {
  car: [4.5, 1.8], van: [5.2, 2.0], truck: [10, 2.5], bus: [12, 2.55], motorcycle: [2.2, 0.8], motorbike: [2.2, 0.8],
  bicycle: [1.8, 0.6], person: [0.6, 0.6], pedestrian: [0.6, 0.6], trailer: [8, 2.5],
};
const sizeOf = (cls: string): [number, number] => SIZE[cls.toLowerCase()] ?? SIZE.car;

type M3 = number[];                                        // 3x3, row-major
const mul = (a: M3, b: M3): M3 => {
  const o = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) o[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
  return o;
};
const inv = (m: M3): M3 => {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((v) => v / det);
};
const apply = (m: M3, x: number, y: number): [number, number] => {
  const w = m[6] * x + m[7] * y + m[8];
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
};

/** The sample in force at time t: the only one for a fixed camera, else the nearest. */
function sampleAt(v: GroundView, t: number): GroundSample {
  const s = v.samples;
  if (v.fixed || s.length === 1) return s[0];
  let lo = 0, hi = s.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (s[mid].t <= t) lo = mid; else hi = mid; }
  return Math.abs(s[lo].t - t) <= Math.abs(s[hi].t - t) ? s[lo] : s[hi];
}

/** Where a box touches the road (the middle of its bottom edge) in metres, and the direction away from the camera
 *  there (up the picture), or null above the drawn part of the frame. */
function contact(s: GroundSample, b: number[], W: number, H: number): { e: number; n: number; ue: number; un: number; cut: boolean } | null {
  const x = (b[0] + b[2]) / 2, y = b[3];
  if (y <= s.top_row) return null;
  const cut = y >= H - 2 || b[0] <= 1 || b[2] >= W - 1;      // cut by the frame's edge: where it touches the road is not seen
  const [e, n] = apply(s.G, x, y);
  const [e2, n2] = apply(s.G, x, Math.max(s.top_row + 0.5, y - 4));
  const l = Math.hypot(e2 - e, n2 - n);
  if (!isFinite(e) || !isFinite(n) || !l) return null;
  return { e, n, ue: (e2 - e) / l, un: (n2 - n) / l, cut };
}

const VERT = `attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }`;
// Each screen pixel back to the frame: screen -> metres -> frame pixel (inverse ground homography), then sampled.
const FRAG = `precision highp float;
uniform mat3 uM; uniform vec2 uFrame; uniform float uTop; uniform float uH; uniform float uSign; uniform sampler2D uTex;
void main() {
  vec3 q = uM * vec3(gl_FragCoord.x, uH - gl_FragCoord.y, 1.0);
  if (q.z * uSign <= 0.0) { gl_FragColor = vec4(0.0); return; }
  vec2 f = q.xy / q.z;
  if (f.x < 0.0 || f.y < uTop || f.x > uFrame.x || f.y > uFrame.y) { gl_FragColor = vec4(0.0); return; }
  gl_FragColor = vec4(texture2D(uTex, f / uFrame).rgb, 1.0);
}`;

function setupGL(c: HTMLCanvasElement) {
  const gl = c.getContext("webgl", { premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!; gl.shaderSource(s, src); gl.compileShader(s); return s;
  };
  const prog = gl.createProgram()!;
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  for (const [k, v] of [[gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
    [gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  const u = (n: string) => gl.getUniformLocation(prog, n);
  return { gl, uM: u("uM"), uFrame: u("uFrame"), uTop: u("uTop"), uH: u("uH"), uSign: u("uSign") };
}

/** A round length for the scale bar: about a fifth of the view's width. */
function niceLength(m: number) {
  const p = Math.pow(10, Math.floor(Math.log10(m)));
  return [1, 2, 5, 10].map((k) => k * p).filter((v) => v <= m).pop() ?? p;
}

export default function BirdsEye({ view, now, source, events, sel, onPick, lookOf, boxAt, aspect }: {
  view: GroundView; now: number; source: () => HTMLVideoElement | HTMLImageElement | null; events: Ev[];
  sel: number | null; onPick: (id: number) => void; lookOf: (e: any) => Look; boxAt: (e: any, t: number) => number[] | null;
  aspect: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const glr = useRef<ReturnType<typeof setupGL> | null>(null);
  const [size, setSize] = useState({ w: 640, h: 360 });
  const [northUp, setNorthUp] = useState(() => { try { return localStorage.getItem("twb.bev.north") === "1"; } catch { return false; } });
  const [zoom, setZoom] = useState({ k: 1, dx: 0, dy: 0 });              // on top of the fit: scale and pan in CSS px
  const [pan, setPan] = useState<{ x: number; y: number; dx: number; dy: number } | null>(null);
  const [noGL, setNoGL] = useState(false);
  const [, redraw] = useState(0);
  const W = view.frame_w, H = view.frame_h;
  const cur = sampleAt(view, now);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => { try { localStorage.setItem("twb.bev.north", northUp ? "1" : "0"); } catch { /* not kept */ } }, [northUp]);

  // Each event's track on the ground (where it touched the road on every analysed frame).
  const tracks = useMemo(() => {
    const out = new Map<number, { t: number; e: number; n: number; ue: number; un: number; cut: boolean }[]>();
    for (const e of events) {
      const pts = [];
      for (const b of e.boxes) {
        const g = contact(sampleAt(view, b[0]), b.slice(1), W, H);
        if (g && !g.cut) pts.push({ t: b[0], ...g });
      }
      out.set(e.id, pts);
    }
    return out;
  }, [events, view]);

  // The view: rotated so the camera looks up the screen (or north up), fitted to the drawn ground, then zoomed.
  const fit = useMemo(() => {
    const s = view.fixed ? view.samples[0] : cur;
    const [e0, n0] = apply(s.G, W / 2, H), [e1, n1] = apply(s.G, W / 2, (s.top_row + H) / 2);
    const th = northUp ? 0 : Math.PI / 2 - Math.atan2(n1 - n0, e1 - e0);
    const c = Math.cos(th), sn = Math.sin(th);
    const rot = ([e, n]: number[]) => [c * e - sn * n, sn * e + c * n];
    // fitted to where the vehicles drove (most of their tracks) and the near part of the frame; to the whole drawn
    // ground when nothing was tracked (or the camera moves)
    const driven: number[][] = [];
    if (view.fixed) for (const tr of tracks.values()) for (const p of tr) driven.push(rot([p.e, p.n]));
    const near = s.near.map(rot);
    let xs: number[], ys: number[];
    if (driven.length >= 20) {
      const q = (a: number[], f: number) => a[Math.min(a.length - 1, Math.max(0, Math.round(f * (a.length - 1))))];
      const dx = driven.map((p) => p[0]).sort((a, b) => a - b), dy = driven.map((p) => p[1]).sort((a, b) => a - b);
      xs = [q(dx, 0.02), q(dx, 0.98), ...near.map((p) => p[0])];
      ys = [q(dy, 0.02), q(dy, 0.98), ...near.map((p) => p[1])];
    } else {
      const all = [...near, ...s.far.map(rot)];
      xs = all.map((p) => p[0]); ys = all.map((p) => p[1]);
    }
    const minx = Math.min(...xs) - 3, maxx = Math.max(...xs) + 3, miny = Math.min(...ys) - 3, maxy = Math.max(...ys) + 3;
    const k = Math.min(size.w / (maxx - minx || 1), size.h / (maxy - miny || 1)) * 0.94;
    return { th, c, sn, k, mx: (minx + maxx) / 2, my: (miny + maxy) / 2 };
  }, [view, northUp, size, tracks, view.fixed ? 0 : cur.t]);
  useEffect(() => { setZoom({ k: 1, dx: 0, dy: 0 }); }, [view, northUp]);
  // metres east, north -> screen CSS px
  const k = fit.k * zoom.k;
  const V: M3 = useMemo(() => {
    const R: M3 = [fit.c, -fit.sn, 0, fit.sn, fit.c, 0, 0, 0, 1];
    const S: M3 = [k, 0, size.w / 2 + zoom.dx - k * fit.mx, 0, -k, size.h / 2 + zoom.dy + k * fit.my, 0, 0, 1];
    return mul(S, R);
  }, [fit, k, zoom, size]);
  const toS = (e: number, n: number) => apply(V, e, n);

  // Draw the ground: every animation frame while the video plays, and on every new frame when paused.
  const draw = () => {
    const cv = canvas.current, src = source();
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.round(size.w * dpr), ch = Math.round(size.h * dpr);
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    if (!glr.current) { glr.current = setupGL(cv); if (!glr.current) { setNoGL(true); return; } }
    const { gl, uM, uFrame, uTop, uH, uSign } = glr.current;
    gl.viewport(0, 0, cw, ch);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!src) return;
    const ready = src instanceof HTMLVideoElement ? src.readyState >= 2 : src.complete && src.naturalWidth > 0;
    if (!ready) return;
    try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); } catch { return; }
    const D: M3 = [dpr, 0, 0, 0, dpr, 0, 0, 0, 1];
    const toFrame = mul(inv(cur.G), inv(mul(D, V)));                    // device px -> frame px
    const Gi = inv(cur.G);
    const [e0, n0] = apply(cur.G, W / 2, H - 1);
    const sign = Math.sign(Gi[6] * e0 + Gi[7] * n0 + Gi[8]) || 1;
    // the frame's own pixel size may differ from the source (the fallback's frames are scaled): sample by fraction
    gl.uniformMatrix3fv(uM, false, new Float32Array([0, 1, 2].flatMap((c) => [0, 1, 2].map((r) => toFrame[r * 3 + c]))));
    gl.uniform2f(uFrame, W, H); gl.uniform1f(uTop, cur.top_row); gl.uniform1f(uH, ch); gl.uniform1f(uSign, sign);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };
  useEffect(draw);
  useEffect(() => {
    const src = source();
    if (!src) return;
    const on = () => redraw((x) => x + 1);
    const evs = src instanceof HTMLVideoElement ? ["seeked", "loadeddata", "canplay"] : ["load"];
    evs.forEach((n) => src.addEventListener(n, on));
    return () => evs.forEach((n) => src.removeEventListener(n, on));
  });

  /** Where the event goes at time t (unit vector, metres) and how fast (m/s), from its track a second around t. */
  const motion = (id: number, t: number) => {
    const tr = tracks.get(id) ?? [];
    const a = tr.filter((p) => p.t >= t - 0.6 && p.t <= t + 0.6);
    if (a.length >= 2) {
      const p = a[0], q = a[a.length - 1], d = Math.hypot(q.e - p.e, q.n - p.n);
      if (d >= 1.0 && q.t > p.t) return { he: (q.e - p.e) / d, hn: (q.n - p.n) / d, v: d / (q.t - p.t) };
    }
    if (tr.length >= 2) {
      const p = tr[0], q = tr[tr.length - 1], d = Math.hypot(q.e - p.e, q.n - p.n);
      if (d >= 3) return { he: (q.e - p.e) / d, hn: (q.n - p.n) / d, v: a.length >= 2 ? 0 : null };
    }
    return null;
  };

  const shapes = events.map((e) => {
    const b = boxAt(e, now);
    if (!b) return null;
    const g = contact(cur, b, W, H);
    if (!g) return null;
    const mo = g.cut ? null : motion(e.id, now);
    const he = mo?.he ?? g.ue, hn = mo?.hn ?? g.un;                       // standing still with no track: along the view
    const [L, Wd] = sizeOf(e.cls);
    const cosp = Math.abs(he * g.ue + hn * g.un), sinp = Math.sqrt(Math.max(0, 1 - cosp * cosp));
    const back = (L / 2) * cosp + (Wd / 2) * sinp;                          // the box's bottom is the side nearest the camera
    const ce = g.e + g.ue * back, cn = g.n + g.un * back;
    const pe = -hn, pn = he;
    const corners = [[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]]
      .map(([a, w]) => toS(ce + he * a + pe * w, cn + hn * a + pn * w));
    const nose = toS(ce + he * (L / 2 + 0.2), cn + hn * (L / 2 + 0.2));
    const tr = (tracks.get(e.id) ?? []).filter((p) => p.t <= now + 0.01);
    const trail = tr.map((p) => toS(p.e, p.n));
    return { e, corners, nose, trail, centre: toS(ce, cn), kmh: mo?.v != null ? mo.v * 3.6 : null, moving: !!mo };
  }).filter(Boolean) as { e: Ev; corners: number[][]; nose: number[]; trail: number[][]; centre: number[]; kmh: number | null; moving: boolean }[];

  const whole = [...cur.far.slice(0, 2), ...cur.near.slice(2)].map(([e, n]) => toS(e, n));
  const farEdge = cur.far.slice(0, 2).map(([e, n]) => toS(e, n));
  const bar = niceLength(size.w / 5 / k);
  const col = view.registered ? "var(--human)" : "var(--machine)";

  const onWheel = (ev: React.WheelEvent) => {
    const r = wrap.current!.getBoundingClientRect();
    const x = ev.clientX - r.left - size.w / 2, y = ev.clientY - r.top - size.h / 2;
    const f = Math.exp(-ev.deltaY * 0.0015);
    setZoom((z) => {
      const nk = Math.max(0.5, Math.min(40, z.k * f)), g = nk / z.k;
      return { k: nk, dx: x - (x - z.dx) * g, dy: y - (y - z.dy) * g };
    });
  };
  useEffect(() => {                                       // the page must not scroll while zooming the view
    const el = wrap.current;
    if (!el) return;
    const stop = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", stop, { passive: false });
    return () => el.removeEventListener("wheel", stop);
  }, []);

  return (
    <div ref={wrap} className="stage bev" style={{ aspectRatio: aspect, maxHeight: "56vh", background: "#0b0f14", cursor: pan ? "grabbing" : "grab" }}
      onWheel={onWheel}
      onPointerDown={(ev) => { if (ev.button === 0) { setPan({ x: ev.clientX, y: ev.clientY, dx: zoom.dx, dy: zoom.dy }); (ev.target as Element).setPointerCapture?.(ev.pointerId); } }}
      onPointerMove={(ev) => { if (pan) setZoom((z) => ({ ...z, dx: pan.dx + ev.clientX - pan.x, dy: pan.dy + ev.clientY - pan.y })); }}
      onPointerUp={() => setPan(null)}>
      <canvas ref={canvas} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
      <svg viewBox={`0 0 ${size.w} ${size.h}`} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
        <polygon points={whole.map((p) => p.join(",")).join(" ")} fill="none" stroke={col} strokeOpacity={0.7} strokeWidth={1.5} />
        <line x1={farEdge[0][0]} y1={farEdge[0][1]} x2={farEdge[1][0]} y2={farEdge[1][1]} stroke="var(--warn)" strokeWidth={2.5} strokeOpacity={0.8} />
        {shapes.map(({ e, trail }) => trail.length > 1 && <polyline key={`t${e.id}`} points={trail.map((p) => p.join(",")).join(" ")} fill="none"
          stroke={lookOf(e).stroke} strokeOpacity={e.id === sel ? 0.95 : 0.55} strokeWidth={e.id === sel ? 2.5 : 1.5} strokeLinejoin="round" />)}
        {shapes.map(({ e, corners, nose, centre, kmh, moving }) => {
          const lk = lookOf(e), on = e.id === sel;
          return <g key={e.id} style={{ cursor: "pointer" }} onPointerDown={(ev) => { ev.stopPropagation(); onPick(e.id); }}>
            <polygon points={corners.map((p) => p.join(",")).join(" ")} fill={lk.stroke} fillOpacity={on ? 0.55 : 0.32}
              stroke={on ? "#fff" : lk.stroke} strokeWidth={on ? 2 : 1.25} strokeDasharray={lk.dash || undefined} />
            {moving && <line x1={centre[0]} y1={centre[1]} x2={nose[0]} y2={nose[1]} stroke={on ? "#fff" : lk.stroke} strokeWidth={1.5} />}
            <text x={centre[0] + 8} y={centre[1] - 6} fontSize={11} fill={lk.stroke} paintOrder="stroke" stroke="#000" strokeWidth={3}>
              {e.key}{kmh != null ? ` · ${Math.round(kmh)} km/h` : ""}</text>
          </g>;
        })}
        <g transform={`translate(12 ${size.h - 14})`}>
          <line x1={0} y1={0} x2={bar * k} y2={0} stroke="#fff" strokeWidth={2} />
          {[0, bar * k].map((x) => <line key={x} x1={x} y1={-4} x2={x} y2={4} stroke="#fff" strokeWidth={2} />)}
          <text x={bar * k / 2} y={-7} fontSize={11} fill="#fff" textAnchor="middle" paintOrder="stroke" stroke="#000" strokeWidth={3}>{bar} m</text>
        </g>
        <g transform={`translate(${size.w - 22} 24) rotate(${(-fit.th * 180) / Math.PI})`}>
          <circle r={13} fill="rgba(0,0,0,0.45)" />
          <path d="M 0 -10 L 5 4 L 0 1 L -5 4 Z" fill="#fff" />
          <text y={-14} fontSize={10} fill="#fff" textAnchor="middle" paintOrder="stroke" stroke="#000" strokeWidth={3}>N</text>
        </g>
      </svg>
      <div className="bev-tools">
        <button onClick={() => setNorthUp(!northUp)} title="Turn the view">{northUp ? "North up" : "Camera up"}</button>
        <button onClick={() => setZoom({ k: 1, dx: 0, dy: 0 })} title="Show the whole ground the camera sees">Fit</button>
      </div>
      {noGL && <div className="stage-note quiet">This browser cannot draw the road from above (no WebGL).</div>}
      {!view.registered && <div className="bev-note">{view.how === "flight-data" ? "Placed by flight data only: positions can be metres off."
        : "The points on the map fit loosely: positions can be metres off."}</div>}
    </div>
  );
}
