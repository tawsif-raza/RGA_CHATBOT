import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produces .next/standalone: a self-contained server bundle with only the
  // production node_modules it actually needs traced in, instead of the
  // full node_modules tree - what frontend-next/Dockerfile's final stage copies.
  output: "standalone",
};

export default nextConfig;
