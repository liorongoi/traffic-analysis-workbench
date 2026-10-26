import { useRoute } from "./router";
import Logo from "./Logo";
import Missions from "./pages/Missions";
import Wizard from "./pages/Wizard";
import MissionPage from "./pages/Mission";
import Investigate from "./pages/Investigate";
import RegionEditor from "./pages/RegionEditor";

export default function App() {
  const route = useRoute();
  const [section, id, sub, extra] = route;
  let body;
  if (section === "new") body = <Wizard />;
  else if (section === "m" && id && sub === "i" && extra) body = <Investigate missionId={id} versionId={Number(extra)} />;
  else if (section === "m" && id && sub === "regions" && extra) body = <RegionEditor missionId={id} viewId={Number(extra)} />;
  else if (section === "m" && id) body = <MissionPage missionId={id} tab={sub ?? "overview"} extra={extra} />;
  else body = <Missions />;

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="top">
        <a className="brand" href="#/" aria-label="LanePulse home">
          <Logo />
          <span className="brand-wordmark"><strong>LanePulse</strong><small>OFFLINE TRAFFIC INTELLIGENCE</small></span>
        </a>
        <span className="spacer" />
        <nav className="top-nav" aria-label="Main navigation"><a href="#/">Missions</a></nav>
      </header>
      <main id="main-content">{body}</main>
    </>
  );
}