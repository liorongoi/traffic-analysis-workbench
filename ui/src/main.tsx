import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import AccessGate from "./components/AccessGate";
// Fonts are bundled with the app, so they show on a workstation with no internet.
import "@fontsource/chakra-petch/latin-500.css";
import "@fontsource/chakra-petch/latin-600.css";
import "@fontsource/chakra-petch/latin-700.css";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "./styles.css";
import "./fit.css";
import "./look2027.css";
// The 2027 look, while it is a proposal: ?look=2027 in the address, or remembered in this browser.
try {
  const q = new URLSearchParams(location.search).get("look");
  if (q !== null) localStorage.setItem("lp-look", q);
  const look = localStorage.getItem("lp-look");
  if (look) document.documentElement.dataset.look = look;
} catch { /* storage blocked: keep the current look */ }

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AccessGate>
      <App />
    </AccessGate>
  </StrictMode>,
);
