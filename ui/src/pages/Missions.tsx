import { useEffect, useState } from "react";
import { api } from "../api";

type M = { id: string; name: string; input_type: string; mode: string; processing: string; monitoring: boolean; created_at: string;
  status: Record<string, number>; job: { state: string; done: number; total: number } | null };

export default function Missions() {
  const [list, setList] = useState<M[] | null>(null);
  useEffect(() => { api("/missions").then((r) => setList(r.missions)); }, []);
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Missions</h1>
        <a className="btn primary" href="#/new"><button className="primary">New mission</button></a>
      </div>
      {list === null && <p className="muted">Loading…</p>}
      {list?.length === 0 && <div className="card"><p>No missions yet.</p><p className="muted">A mission reads one source folder of images or videos, analyses them in one mode and exports to one output folder.</p></div>}
      {list && list.length > 0 && (
        <div className="tablewrap">
          <table className="t">
            <thead><tr><th>Name</th><th>Input</th><th>Mode</th><th>Processing</th><th className="num">Files</th><th className="num">Queued</th><th>Activity</th><th>Created (UTC)</th></tr></thead>
            <tbody>
              {list.map((m) => (
                <tr key={m.id} className="click" onClick={() => (window.location.hash = `#/m/${m.id}`)}>
                  <td><b className="bidi">{m.name}</b></td>
                  <td>{m.input_type}</td>
                  <td>{m.mode.replace(/_/g, " ")}</td>
                  <td>{m.processing === "continuous" ? `continuous, monitoring ${m.monitoring ? "on" : "off"}` : "one-time"}</td>
                  <td className="num">{m.status.source_versions}</td>
                  <td className="num">{m.status.queued_items}</td>
                  <td>{m.job && ["running", "paused"].includes(m.job.state) ? `${m.job.state} ${m.job.done}/${m.job.total}` : "idle"}</td>
                  <td className="mono">{m.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
