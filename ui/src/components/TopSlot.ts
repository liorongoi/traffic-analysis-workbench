import { useEffect, useState } from "react";

/** The place in the application bar where a page can put its own title and tabs, so the page needs no second bar
 *  (one bar on mission pages: the work below gets the room). Null until the bar is on screen. */
export function useTopSlot(): HTMLElement | null {
  const [el, setEl] = useState<HTMLElement | null>(null);
  useEffect(() => { setEl(document.getElementById("top-slot")); }, []);
  return el;
}
