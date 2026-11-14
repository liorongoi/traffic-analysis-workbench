import { useEffect, useState } from "react";
import { useRoute } from "./router";
import { api } from "./api";
import Logo from "./Logo";
import JumpBar from "./components/JumpBar";
import Missions from "./pages/Missions";
import Wizard from "./pages/Wizard";
import QuickStart from "./pages/QuickStart";
import MissionPage from "./pages/Mission";
import Investigate from "./pages/Investigate";
import RegionEditor from "./pages/RegionEditor";
import NameFormats from "./pages/NameFormats";
import Workers from "./pages/Workers";
import GeoPlace from "./pages/GeoPlace";

export default function App() {
  const route = useRoute();
  const [section, id, sub, extra] = route;
  let body;
  if (section === "new" && id === "full") body = <Wizard />;
  else if (section === "new") body = <QuickStart />;
  else if (section === "formats") body = <NameFormats />;
  else if (section === "workers") body = <Workers />;
  else if (section === "m" && id && sub === "i" && extra) body = <Investigate missionId={id} versionId={Number(extra)} />;
  else if (section === "m" && id && sub === "map" && extra) body = <GeoPlace missionId={id} versionId={Number(extra)} />;
  else if (section === "m" && id && sub === "regions" && extra) body = <RegionEditor missionId={id} viewId={Number(extra)} />;
  else if (section === "m" && id) body = <MissionPage missionId={id} tab={sub ?? "overview"} extra={extra} />;
  else body = <Missions />;

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="top">
        <a className="brand" href="#/" aria-label="LanePulse home">
          <Logo />
          <span className="brand-wordmark"><strong>LanePulse</strong><small>Traffic intelligence</small></span>
        </a>
        <nav className="top-nav" aria-label="Main navigation">
          <a href="#/" aria-current={section === "m" ? "true" : section !== "new" && section !== "formats" && section !== "workers" ? "page" : undefined}>Missions</a>
          <a href="#/new" aria-current={section === "new" ? "page" : undefined}>New mission</a>
          <a href="#/formats" aria-current={section === "formats" ? "page" : undefined}>Filename formats</a>
          <a href="#/workers" aria-current={section === "workers" ? "page" : undefined}>Workers</a>
        </nav>
        <div className="top-tools">
          <JumpBar />
          <Status />
          <Clock />
          <Version />
        </div>
      </header>
      <main id="main-content">{body}</main>
    </>
  );
}

const ACCEL: Record<string, string> = { cuda: "GPU ready", directml: "GPU ready", cpu: "Running on CPU" };

/** Where models run, and whether any mission is processing right now. */
function Status() {
  const [text, setText] = useState<{ label: string; busy: boolean } | null>(null);
  useEffect(() => {
    let live = true;
    const load = () => Promise.all([api("/system"), api("/missions")]).then(([sys, ms]) => {
      if (!live) return;
      const busy = (ms.missions as any[]).filter((m) => m.job?.state === "running").length;
      const where = sys.runtime ? ACCEL[sys.runtime.accelerator] ?? sys.runtime.accelerator : "Detector idle";
      setText({ label: busy ? `${busy} processing · ${where}` : where, busy: busy > 0 });
    }).catch(() => { if (live) setText(null); });
    load();
    const t = window.setInterval(load, 15000);
    return () => { live = false; window.clearInterval(t); };
  }, []);
  if (!text) return null;
  return <span className={`status-dot${text.busy ? " busy" : ""}`} role="status"><i aria-hidden="true" />{text.label}</span>;
}

/** Local time and UTC side by side: capture times in this app are in UTC, and people think in local time. */
function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = window.setInterval(() => setNow(new Date()), 15000); return () => window.clearInterval(t); }, []);
  const hm = (d: Date, utc: boolean) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false, ...(utc ? { timeZone: "UTC" } : {}) });
  return <span className="clock" aria-label={`Local time ${hm(now, false)}, UTC ${hm(now, true)}`}>{hm(now, false)} <span>local</span> · {hm(now, true)} <span>UTC</span></span>;
}

/** The build this server runs, so whoever tests can tell which version is on screen. */
function Version() {
  const [v, setV] = useState<string | null>(null);
  useEffect(() => { api("/health").then((h) => setV(h.version)).catch(() => setV(null)); }, []);
  if (!v) return null;
  return <span className="build-version" title={`LanePulse version ${v}`}>v{v}</span>;
}
