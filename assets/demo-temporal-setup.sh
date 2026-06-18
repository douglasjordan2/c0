#!/usr/bin/env bash
# Idempotent setup for the bi-temporal demo (assets/demo-temporal.tape).
# Builds a small, accurate Next.js routing history: Pages Router (2022) -> App Router (2023).
# Run by the tape's hidden block so the GIF renders deterministically.
set -e

names="['nextjs-routing','nextjs-app-router','pages-directory','react-server-components']"
c0 find "MATCH (c:Concept) WHERE c.name IN ${names} DETACH DELETE c RETURN count(c)" >/dev/null 2>&1 || true

c0 add concept "nextjs-routing"          -d "Next.js routing via the Pages Router: each file in pages/ becomes a route." --valid-at 2022-01-01 --force >/dev/null 2>&1
c0 add concept "pages-directory"         -d "The pages/ directory convention; data via getServerSideProps." --valid-at 2022-01-01 --force >/dev/null 2>&1
c0 add concept "nextjs-app-router"       -d "Next.js App Router: the app/ directory built on React Server Components." --valid-at 2023-05-01 --force >/dev/null 2>&1
c0 add concept "react-server-components" -d "Components that render on the server by default; foundation of the App Router." --valid-at 2023-05-01 --force >/dev/null 2>&1

c0 relate "nextjs-routing"   USES "pages-directory"         >/dev/null 2>&1
c0 relate "nextjs-app-router" USES "react-server-components" >/dev/null 2>&1
