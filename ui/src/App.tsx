import { useEffect, useState } from "react";
import { api, post, setSignedOutHandler, User } from "./api";
import { go, useRoute } from "./router";
import Login from "./pages/Login";
import Missions from "./pages/Missions";
import Wizard from "./pages/Wizard";
import MissionPage from "./pages/Mission";
import Investigate from "./pages/Investigate";
import RegionEditor from "./pages/RegionEditor";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);
  const [needsSetup, setNeedsSetup] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const route = useRoute();

  const refresh = async () => {
    const s = await api("/session");
    setUser(s.signed_in ? s.user : null);
    setNeedsSetup(!!s.needs_setup);
    if (!s.signed_in && s.message && s.message !== "Sign in to continue.") setNotice(s.message);
    setChecked(true);
  };
  useEffect(() => {
    setSignedOutHandler((msg) => {
      setUser(null);
      setNotice(msg);
    });
    refresh();
  }, []);

  if (!checked) return <div className="page muted">Loading…</div>;
  if (!user) return <Login needsSetup={needsSetup} notice={notice} onDone={() => { setNotice(null); refresh(); }} />;

  const [section, id, sub, extra] = route;
  let body;
  if (section === "new") body = <Wizard />;
  else if (section === "m" && id && sub === "i" && extra) body = <Investigate missionId={id} versionId={Number(extra)} />;
  else if (section === "m" && id && sub === "regions" && extra) body = <RegionEditor missionId={id} viewId={Number(extra)} />;
  else if (section === "m" && id) body = <MissionPage missionId={id} tab={sub ?? "overview"} />;
  else body = <Missions />;

  return (
    <>
      <header className="top">
        <a className="brand" href="#/">Traffic Analysis Workbench</a>
        <span className="spacer" />
        <span className="muted">Signed in as <b>{user.display_name}</b></span>
        <button onClick={async () => { await post("/logout"); setUser(null); go("/"); }}>Sign out</button>
      </header>
      {body}
    </>
  );
}
