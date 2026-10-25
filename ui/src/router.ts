import { useEffect, useState } from "react";

// Hash routes keep the server simple and work offline: #/missions, #/m/<id>/results, #/m/<id>/i/<vid> ...
export function useRoute(): string[] {
  const parse = () => window.location.hash.replace(/^#\/?/, "").split("?")[0].split("/").filter(Boolean);
  const [parts, setParts] = useState(parse);
  useEffect(() => {
    const on = () => setParts(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return parts;
}

export function go(path: string) {
  window.location.hash = path.startsWith("#") ? path : `#/${path.replace(/^\//, "")}`;
}
