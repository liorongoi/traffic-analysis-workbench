import { useEffect, useMemo, useState } from "react";

// Pick-from-list controls. Every setting in LanePulse is chosen from a menu; typing is only ever an optional extra
// ("Something else…") for the few things no list can hold, such as a name of the operator's own.

type Opt<T> = T | [T, string];
const optValue = <T,>(o: Opt<T>): T => (Array.isArray(o) ? o[0] : o);

/** A menu of preset values. A value saved earlier that is not one of the presets stays shown and selected. */
export function Choice<T extends string | number>({ value, options, onChange, label, placeholder, format, style, disabled }: {
  value: T | "" | null | undefined; options: Opt<T>[]; onChange: (v: T) => void; label: string;
  placeholder?: string; format?: (v: T) => string; style?: React.CSSProperties; disabled?: boolean;
}) {
  const numeric = typeof optValue(options[0]) === "number";
  const vals = options.map((o) => String(optValue(o)));
  const cur = value === null || value === undefined ? "" : String(value);
  const show = (o: Opt<T>) => (Array.isArray(o) ? o[1] : format ? format(o) : String(o));
  return (
    <select aria-label={label} value={cur} disabled={disabled} style={style}
      onChange={(e) => { if (e.target.value !== "") onChange((numeric ? Number(e.target.value) : e.target.value) as T); }}>
      {(placeholder !== undefined || cur === "") && <option value="" disabled={cur !== ""}>{placeholder ?? "Choose…"}</option>}
      {cur !== "" && !vals.includes(cur) && <option value={cur}>{format ? format(value as T) : cur}</option>}
      {options.map((o) => <option key={String(optValue(o))} value={String(optValue(o))}>{show(o)}</option>)}
    </select>
  );
}

const OTHER = "\u0000other";

/** A name chosen from suggestions. The first suggestion is used until another is chosen, so nothing has to be typed;
 *  "Something else…" opens a box for a name of one's own. */
export function NamePick({ value, onChange, suggestions, label, optional, style }: {
  value: string; onChange: (v: string) => void; suggestions: string[]; label: string; optional?: boolean; style?: React.CSSProperties;
}) {
  const list = useMemo(() => [...new Set(suggestions.map((s) => s.trim()).filter(Boolean))], [suggestions]);
  const [own, setOwn] = useState(() => !!value && !list.includes(value));
  // A name is required: start from the first suggestion rather than an empty box.
  useEffect(() => { if (!optional && !own && value === "" && list[0]) onChange(list[0]); }, [optional, own, value, list[0]]);
  return (
    <span className="namepick" style={style}>
      <select aria-label={label} value={own ? OTHER : value}
        onChange={(e) => {
          if (e.target.value === OTHER) { setOwn(true); return; }
          setOwn(false); onChange(e.target.value);
        }}>
        {optional && <option value="">{own ? "No name" : "No name (optional)"}</option>}
        {!optional && value === "" && !own && <option value="" disabled>Choose a name…</option>}
        {!own && value !== "" && !list.includes(value) && <option value={value}>{value}</option>}
        {list.map((s) => <option key={s} value={s}>{s}</option>)}
        <option value={OTHER}>Something else…</option>
      </select>
      {own && <input dir="auto" autoFocus aria-label={`${label}, your own`} value={value} onChange={(e) => onChange(e.target.value)} />}
    </span>
  );
}

/** A time zone in two menus: the area (Europe, Asia, …), then the city. */
export function ZonePick({ zones, value, onChange, extra }: {
  zones: string[]; value: string; onChange: (z: string) => void; extra?: string[];
}) {
  const areas = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const z of zones) {
      if (!z.includes("/") || z.startsWith("Etc/") || z.startsWith("SystemV/")) continue;
      const a = z.split("/")[0];
      if (!m.has(a)) m.set(a, []);
      m.get(a)!.push(z);
    }
    return m;
  }, [zones]);
  const area = value.includes("/") ? value.split("/")[0] : "";
  const [shown, setShown] = useState(area);
  const pickArea = shown || area;
  const cities = areas.get(pickArea) ?? [];
  const city = (z: string) => z.split("/").slice(1).join(" / ").replace(/_/g, " ");
  return (
    <span className="zonepick">
      <select aria-label="Time-zone area" value={pickArea} onChange={(e) => {
        const a = e.target.value; setShown(a);
        if (extra?.includes(a)) { onChange(a); return; }
        const first = areas.get(a)?.[0]; if (first) onChange(first);
      }}>
        {extra?.map((x) => <option key={x} value={x}>{x}</option>)}
        {[...areas.keys()].map((a) => <option key={a} value={a}>{a}</option>)}
      </select>
      {cities.length > 0 && (
        <select aria-label="City" value={cities.includes(value) ? value : ""} onChange={(e) => onChange(e.target.value)}>
          {!cities.includes(value) && <option value="" disabled>Choose a city…</option>}
          {cities.map((z) => <option key={z} value={z}>{city(z)}</option>)}
        </select>
      )}
    </span>
  );
}

/** Times of day every half hour, "00:00" to "23:30", plus "24:00" (midnight at the end of the day) when asked. */
export function halfHours(withMidnight = false): string[] {
  const out: string[] = [];
  for (let h = 0; h < 24; h++) for (const m of ["00", "30"]) out.push(`${String(h).padStart(2, "0")}:${m}`);
  if (withMidnight) out.push("24:00");
  return out;
}

/** "13 Nov 2026" style, for default names. */
export const today = () => new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
export const thisMonth = () => new Date().toLocaleDateString("en-GB", { month: "long", year: "numeric" });
