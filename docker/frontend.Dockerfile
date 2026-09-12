# Frontend build + static serve (Section 19 demo deployment).
FROM node:22-slim AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-fund --no-audit
COPY frontend/ .
RUN npm run build

FROM node:22-slim
WORKDIR /app
RUN npm install --no-fund --no-audit --silent serve@14
COPY --from=build /app/dist ./dist
EXPOSE 3000
# SPA fallback so /history and /models resolve on refresh
CMD ["npx", "serve", "-s", "dist", "-l", "3000"]
