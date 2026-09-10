import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.API_PROXY_TARGET || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      proxy: Object.fromEntries(
        ["/api", "/health", "/ready"].map((path) => [path, { target, changeOrigin: true }]),
      ),
    },
  };
});
