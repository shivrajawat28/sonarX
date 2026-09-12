import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: frontend :5173 -> backend :8000 (Section 19: local dev setup).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
