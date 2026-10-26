import { useEffect, useState } from "react";
import { api, User } from "./api";
import { useRoute } from "./router";
import Logo from "./Logo";
import Missions from "./pages/Missions";
import Wizard from "./pages/Wizard";
import MissionPage from "./pages/Mission";
import Investigate from "./pages/Investigate";
import RegionEditor from "./pages/RegionEditor";

export default function App() {
  const [user, setUser] = useState<User>({ id: 0, username: "local", display_name: "Local operator", role: "admin" });
  const [checked, setChecked] = useState(false);
  const route = useRoute();

  const refresh = async () => {
    const s = await api("/session");
    setUser(s.user);
    setChecked(true);
  };
  useEffect(() => {
    refresh();
  }, []);

  if (!checked) return <div className="page muted">Loading…</div>;
  const [section, id, sub, extra] = route;
  let body;
  if (section === "new") body = <Wizard />;
  else if (section === "m" && id && sub === "i" && extra) body = <Investigate missionId={id} versionId={Number(extra)} />;
  else if (section === "m" && id && sub === "regions" && extra) body = <RegionEditor missionId={id} viewId={Number(extra)} />;
  else if (section === "m" && id) body = <MissionPage missionId={id} tab={sub ?? "overview"} extra={extra} />;
  else body = <Missions />;

  return (
    <>
      <header className="top">
        <a className="brand" href="#/"><Logo /> Lanemark</a>
        <span className="spacer" />
        <span className="muted">Local operator: <b>{user.display_name}</b></span>
      </header>
      {body}
    </>
  );
}
