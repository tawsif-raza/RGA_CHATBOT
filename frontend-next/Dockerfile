# Built from the project root: docker build -f frontend-next/Dockerfile .
# Multi-stage: `deps` installs once, `builder` runs `next build` (which
# needs devDependencies and produces .next/standalone via next.config.ts's
# output: "standalone"), and the final stage only copies the traced
# standalone output - not the full node_modules tree either stage used.

FROM node:20-slim AS deps
WORKDIR /app
COPY frontend-next/package.json frontend-next/package-lock.json ./
RUN npm ci

FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend-next/ .
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

# .next/standalone already contains just the production node_modules Next
# traced as actually required (see next.config.ts's output: "standalone")
# - the public/ and .next/static/ copies below are the two directories
# standalone mode deliberately excludes (meant to be served by a CDN in a
# larger deployment), documented in Next's own output.md as required for
# server.js to serve them itself.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
