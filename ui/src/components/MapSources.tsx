import { useEffect, useState } from "react";
import { api, ApiError, post } from "../api";
import FolderPicker from "./FolderPicker";
import { Choice } from "./Pick";

// Where the map picture and the ground heights come from. Three ways in, the same for both:
//   files on this computer (picked from a folder), an online service (from a short list), or a server on our own
//   network (WMTS, WMS, a {z}/{x}/{y} tile address, ArcGIS REST, or a height service), with an optional sign-in.
// Whatever is fetched is saved as a GeoTIFF in LanePulse's maps folder and then used like a picked file, so an area
// fetched once keeps working with no network.

export type Role = "map" | "terrain";
export type Source = { id: string; name: string; role: Role; kind: string; url: string; layer?: string | null;
  style?: string | null; matrix_set?: string | null; format?: string | null; encoding?: string | null;
  max_zoom?: number | null; attribution?: string; auth?: { type: string; user?: string; password?: string; name?: string; key?: string } };
export type Catalogue = { presets: Source[]; servers: Source[]; kinds: Record<string, string>; encodings: Record<string, string>; maps_folder: string };
type Mode = "file" | "online" | "server" | "none";
type FileRow = { name: string; path: string; geotiff?: boolean };
export type LatLon = [number, number];

const SIDES: [number, string][] = [[250, "250 m across"], [500, "500 m across"], [1000, "1 km across"], [2000, "2 km across"], [5000, "5 km across"], [10000, "10 km across"]];
const DETAIL: [number, string][] = [[0.15, "As sharp as the source has"], [0.3, "About 30 cm a pixel"], [0.6, "About 60 cm a pixel"], [1.2, "About 1.2 m a pixel"], [2.5, "About 2.5 m a pixel"]];

export default function MapSource({ role, cat, path, onPath, centres, onCatalogue }: {
  role: Role; cat: Catalogue | null; path: string | null; onPath: (p: string | null) => void;
  centres: { label: string; ll: LatLon | null }[]; onCatalogue: () => void;
}) {
  const terrain = role === "terrain";
  const inMaps = !!(path && cat && path.startsWith(cat.maps_folder));
  const [mode, setMode] = useState<Mode>(terrain && !path ? "none" : "file");
  const [folder, setFolder] = useState<string[]>([]);
  const [files, setFiles] = useState<FileRow[]>([]);
  const [sourceId, setSourceId] = useState("");
  const [centre, setCentre] = useState(0);
  const [typed, setTyped] = useState("");
  const [side, setSide] = useState(1000);
  const [detail, setDetail] = useState(0.3);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [editing, setEditing] = useState<Partial<Source> | null>(null);

  // A path chosen elsewhere (a saved camera) shows its folder.
  useEffect(() => { if (path) { setFolder([path.replace(/[\\/][^\\/]*$/, "")]); if (mode === "none") setMode("file"); } }, [path]);
  useEffect(() => {
    if (!folder[0]) { setFiles([]); return; }
    api(`/${terrain ? "terrain" : "map"}-files?folder=${encodeURIComponent(folder[0])}`).then((r) => setFiles(r.files))
      .catch((e) => { setFiles([]); setMsg({ kind: "err", text: (e as ApiError).message }); });
  }, [folder[0], path]);

  const presets = (cat?.presets ?? []).filter((s) => s.role === role);
  const servers = (cat?.servers ?? []).filter((s) => s.role === role);
  const list = mode === "online" ? presets : servers;
  useEffect(() => { if (!list.find((s) => s.id === sourceId)) setSourceId(list[0]?.id ?? ""); }, [mode, cat]);
  const usable = centres.map((c, i) => [i, c] as const).filter(([, c]) => c.ll);
  useEffect(() => { if (!centres[centre]?.ll && usable[0]) setCentre(usable[0][0]); }, [centres.map((c) => !!c.ll).join()]);

  const typedLL = (): LatLon | null => {
    const m = typed.match(/(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)/);
    return m ? [Number(m[1]), Number(m[2])] : null;
  };
  const where: LatLon | null = centre === -1 ? typedLL() : centres[centre]?.ll ?? null;

  const get = async () => {
    if (!where || !sourceId) return;
    setBusy(true); setMsg(null);
    try {
      const r = await post("/map-sources/fetch", { source_id: sourceId, lat: where[0], lon: where[1], side_m: side, m_per_px: terrain ? 10 : detail });
      setFolder([cat!.maps_folder]);
      onPath(r.path);
      setMsg({ kind: "ok", text: `Saved on this computer as ${r.name} (${r.width}×${r.height}, about ${r.m_per_px} m a pixel)${r.attribution ? `. Source: ${r.attribution}` : ""}. It works with no network from now on.` });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
    setBusy(false);
  };

  const remove = async (id: string) => {
    try { await api(`/map-sources/servers/${id}`, { method: "DELETE" }); onCatalogue(); } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };

  const tabs: [Mode, string, string][] = [
    ...(terrain ? [["none", "None", "Flat ground"] as [Mode, string, string]] : []),
    ["file", "Files", "On this computer"], ["online", "Online service", "Needs the internet"], ["server", "Our server", "WMTS, WMS or REST"],
  ];
  const chosen = list.find((s) => s.id === sourceId);
  return (
    <div className="grid mapsrc" style={{ gap: 8 }}>
      <div className="seg mapsrc-tabs" role="tablist" aria-label={terrain ? "Where the ground heights come from" : "Where the map comes from"}>
        {tabs.map(([k, a, b]) => <button key={k} role="tab" aria-selected={mode === k} className={mode === k ? "on" : ""}
          onClick={() => { setMode(k); setMsg(null); if (k === "none") onPath(null); }}><span>{a}</span><small>{b}</small></button>)}
      </div>

      {mode === "none" && <div className="muted small">The ground is taken as flat. Add ground heights (a terrain model, DTM) to see each point's height,
        to be told when the points are on a slope, and to shade hills on the map.</div>}

      {mode === "file" && <>
        <FolderPicker label={terrain ? "Folder with the terrain file" : "Folder with the map file"} value={folder} onChange={setFolder}
          placeholder={terrain ? "Folder with a DTM GeoTIFF or SRTM .hgt files" : "Folder with an orthophoto or satellite picture"} />
        {files.length > 0 && <select aria-label={terrain ? "Terrain file" : "Map file"} value={path ?? ""} onChange={(e) => onPath(e.target.value || null)}>
          <option value="">{terrain ? "Pick a terrain file…" : "Pick a map file…"}</option>
          {files.map((f) => <option key={f.path} value={f.path}>{f.name}{f.geotiff === false ? " (needs its corners)" : ""}</option>)}
        </select>}
        {folder[0] && files.length === 0 && <div className="muted small">{terrain ? "No GeoTIFF or .hgt file in that folder." : "No GeoTIFF, JPEG or PNG in that folder."}</div>}
        {cat && !inMaps && <button className="linkish small" style={{ justifySelf: "start" }} onClick={() => setFolder([cat.maps_folder])}>
          Show what was fetched before (LanePulse's maps folder)</button>}
      </>}

      {(mode === "online" || mode === "server") && <>
        <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
          {list.length > 0 ? <Choice label={terrain ? "Terrain source" : "Map source"} value={sourceId} style={{ flex: "1 1 260px" }}
            options={list.map((s) => [s.id, mode === "server" ? `${s.name} · ${cat?.kinds[s.kind] ?? s.kind}${s.layer ? ` · ${s.layer}` : ""}` : s.name] as [string, string])}
            onChange={setSourceId} />
            : <div className="muted small" style={{ flex: "1 1 260px" }}>{mode === "server" ? "No server of ours is set up yet." : "No online service for this."}</div>}
          {mode === "server" && <button onClick={() => setEditing({ role, kind: terrain ? "elevation-api" : "wmts", auth: { type: "none" } })}>Add a server…</button>}
          {mode === "server" && chosen && <button onClick={() => setEditing(chosen)}>Change</button>}
          {mode === "server" && chosen && <button onClick={() => remove(chosen.id)} aria-label="Remove this server">Remove</button>}
        </div>
        {mode === "online" && <div className="muted small">Only when this computer is online. What is fetched is kept, so the same area works offline afterwards.</div>}
        {editing && <ServerForm cat={cat!} start={editing} onClose={() => setEditing(null)}
          onSaved={(s) => { setEditing(null); onCatalogue(); setSourceId(s.id); }} />}
        {list.length > 0 && <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <Choice label="Area centred on" value={centre} style={{ flex: "1 1 200px" }}
            options={[...usable.map(([i, c]) => [i, c.label] as [number, string]), [-1, "Coordinates I type…"]]} onChange={setCentre} />
          {centre === -1 && <input aria-label="Latitude, longitude" value={typed} onChange={(e) => setTyped(e.target.value)}
            placeholder="32.0765, 34.7835" style={{ flex: "1 1 160px" }} />}
          <Choice label="Area size" value={side} options={SIDES} onChange={setSide} style={{ flex: "0 1 150px" }} />
          {!terrain && <Choice label="Detail" value={detail} options={DETAIL} onChange={setDetail} style={{ flex: "0 1 210px" }} />}
          <button className="primary" disabled={busy || !where || !sourceId} onClick={get}>
            {busy ? "Fetching…" : terrain ? "Get the heights" : "Get the map"}</button>
        </div>}
        {where && list.length > 0 && <div className="muted small mono">{where[0].toFixed(5)}°, {where[1].toFixed(5)}°</div>}
      </>}
      {msg && <div className={`msg ${msg.kind} small`}>{msg.text}</div>}
    </div>
  );
}

const AUTH: [string, string][] = [["none", "No sign-in"], ["basic", "User name and password"], ["header", "API key in a header"], ["query", "API key in the address"]];

/** Add or change a server on our network: kind, address, sign-in; connect to list its layers, then pick one. */
function ServerForm({ cat, start, onClose, onSaved }: { cat: Catalogue; start: Partial<Source>; onClose: () => void; onSaved: (s: Source) => void }) {
  const [s, setS] = useState<Partial<Source>>({ auth: { type: "none" }, ...start });
  const [layers, setLayers] = useState<{ name: string; title: string; ok?: boolean; matrix_sets?: string[] }[] | null>(null);
  const [formats, setFormats] = useState<string[]>([]);
  const [note, setNote] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (p: Partial<Source>) => { setS((o) => ({ ...o, ...p })); };
  const auth = s.auth ?? { type: "none" };
  const terrain = s.role === "terrain";
  const needsLayer = s.kind === "wms" || s.kind === "wmts";
  const kinds = Object.entries(cat.kinds).filter(([k]) => terrain || k !== "elevation-api") as [string, string][];
  const hint: Record<string, string> = {
    wmts: "The server's address, or its WMTSCapabilities.xml, e.g. http://maps.local:8080/geoserver/gwc/service/wmts",
    wms: "The server's WMS address, e.g. http://maps.local:8080/geoserver/wms",
    xyz: "A tile address with {z}, {x} and {y}, e.g. http://maps.local/tiles/{z}/{x}/{y}.png",
    arcgis: "A MapServer or ImageServer address, e.g. https://gis.local/arcgis/rest/services/Ortho/MapServer",
    "elevation-api": "An Open-Elevation style lookup, e.g. http://heights.local:8080/api/v1/lookup",
  };

  const connect = async () => {
    setBusy(true); setNote(null);
    try {
      const r = await post("/map-sources/connect", { server: { ...s, name: s.name || "test" } });
      setLayers(r.layers ?? []); setFormats(r.formats ?? []);
      if (needsLayer && !(r.layers ?? []).length) setNote({ kind: "err", text: "The server answered but lists no layers." });
      else setNote({ kind: "ok", text: needsLayer ? `Connected: ${r.layers.length} layers.` : r.title ?? "Connected." });
      if (needsLayer && !s.layer && r.layers?.length) set({ layer: (r.layers.find((l: { ok?: boolean }) => l.ok !== false) ?? r.layers[0]).name });
    } catch (e) { setNote({ kind: "err", text: (e as ApiError).message }); }
    setBusy(false);
  };
  const save = async () => {
    try { const r = await post("/map-sources/servers", s); onSaved(r.server); }
    catch (e) { setNote({ kind: "err", text: (e as ApiError).message }); }
  };
  const tiffs = formats.filter((f) => /tif/i.test(f));
  return (
    <div className="card grid serverform" style={{ gap: 8 }}>
      <strong className="small">{s.id ? "Change the server" : "A server on our network"}</strong>
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        <label className="small grow">Name<input value={s.name ?? ""} onChange={(e) => set({ name: e.target.value })} placeholder="City orthophoto 2026" /></label>
        <label className="small">Gives<Choice label="Gives" value={s.role ?? "map"} options={[["map", "The map picture"], ["terrain", "Ground heights"]]}
          onChange={(v) => set({ role: v as Role, kind: v === "map" && s.kind === "elevation-api" ? "wmts" : s.kind })} /></label>
        <label className="small">Kind<Choice label="Kind of server" value={s.kind ?? "wmts"} options={kinds} onChange={(v) => { set({ kind: v, layer: null }); setLayers(null); }} /></label>
      </div>
      <label className="small">Address<input value={s.url ?? ""} onChange={(e) => set({ url: e.target.value })} placeholder={hint[s.kind ?? "wmts"]} /></label>
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        <label className="small">Sign-in<Choice label="Sign-in" value={auth.type} options={AUTH} onChange={(v) => set({ auth: { ...auth, type: v } })} /></label>
        {auth.type === "basic" && <>
          <label className="small grow">User name<input value={auth.user ?? ""} onChange={(e) => set({ auth: { ...auth, user: e.target.value } })} autoComplete="off" /></label>
          <label className="small grow">Password<input type="password" value={auth.password ?? ""} onChange={(e) => set({ auth: { ...auth, password: e.target.value } })} autoComplete="new-password" /></label>
        </>}
        {(auth.type === "header" || auth.type === "query") && <>
          <label className="small">{auth.type === "header" ? "Header" : "Parameter"}
            <Choice label="Key name" value={auth.name ?? (auth.type === "header" ? "X-API-Key" : "apikey")}
              options={auth.type === "header" ? ["X-API-Key", "Authorization", "apikey", "X-Auth-Token"] : ["apikey", "api_key", "key", "token", "access_token"]}
              onChange={(v) => set({ auth: { ...auth, name: v } })} /></label>
          <label className="small grow">Key<input type="password" value={auth.key ?? ""} onChange={(e) => set({ auth: { ...auth, key: e.target.value } })} autoComplete="new-password" /></label>
        </>}
      </div>
      {terrain && s.kind === "xyz" && <label className="small">Heights are stored as
        <Choice label="Height encoding" value={s.encoding ?? ""} placeholder="Pick how the tiles hold heights…" options={Object.entries(cat.encodings) as [string, string][]} onChange={(v) => set({ encoding: v })} /></label>}
      {layers && needsLayer && <label className="small">Layer
        <Choice label="Layer" value={s.layer ?? ""} placeholder="Pick a layer…"
          options={layers.map((l) => [l.name, `${l.title}${l.title !== l.name ? ` (${l.name})` : ""}${l.ok === false ? " · no usable grid" : ""}`] as [string, string])}
          onChange={(v) => set({ layer: v })} /></label>}
      {layers && s.kind === "wms" && terrain && tiffs.length > 0 && <label className="small">Format
        <Choice label="Format" value={s.format ?? tiffs[0]} options={tiffs} onChange={(v) => set({ format: v })} /></label>}
      {note && <div className={`msg ${note.kind} small`}>{note.text}</div>}
      <div className="row" style={{ gap: 8, justifyContent: "flex-end" }}>
        <button onClick={onClose}>Cancel</button>
        <button disabled={busy || !s.url} onClick={connect}>{busy ? "Connecting…" : "Connect"}</button>
        <button className="primary" disabled={!s.name || !s.url || (needsLayer && !s.layer)} onClick={save}>Save the server</button>
      </div>
      <div className="muted small">Passwords and keys are kept on the computer LanePulse runs on and are never shown again.</div>
    </div>
  );
}
