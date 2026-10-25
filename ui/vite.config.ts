import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built on the staging machine and served by the backend; nothing is fetched at run time.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/trafficwb/web/static", emptyOutDir: true, sourcemap: false },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
