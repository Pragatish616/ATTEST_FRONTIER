# DEPLOY.md — ATTEST production deployment

Backend on Railway (Docker, single replica), dashboard on Vercel (static),
trace store on Supabase. Written 30 July 2026 against the state that is
actually deployed, not a plan.

Read this in order. Steps 0–2 must happen before step 3, because the Vercel
origin does not exist until Vercel has deployed once, and the backend's CORS
allowlist needs that origin.

---

## Current live state

| Piece | Value |
|---|---|
| Railway project | `attest` (workspace `pragatish616's Projects`) |
| Railway service | `attest-api`, environment `production` |
| Backend URL | `https://attest-api-production.up.railway.app` |
| Deploys from | GitHub `Pragatish616/ATTEST_FRONTIER`, branch `main` |
| Supabase project | `ATTEST` (`fcnepzhildbupbgqalqk`), region ap-south-1 |
| Migrations applied | `001_init`, `002_rls` (RLS enabled **and** forced) |
| Auth | `ATTEST_API_KEY` set — every route except `/v1/health` needs a bearer token |
| Dashboard | not yet deployed — that is step 3 |

---

## Step 0 — Point your local repo at the right remote

**This is the trap.** Your local `origin` is `Pragatish616/attest`, but Railway
deploys from `Pragatish616/ATTEST_FRONTIER`. Pushing to `origin` right now
changes nothing in production.

```bash
git remote -v                      # confirm: origin = Pragatish616/attest
git remote add frontier https://github.com/Pragatish616/ATTEST_FRONTIER.git
git fetch frontier
```

From here on, `git push frontier main` is the command that triggers a deploy.

Decide now whether you want two repos at all. `Pragatish616/attest` is public
and still carries `CLAUDE.md`, `HANDOFF.md`, and an outdated README. Archive it
or make it private before judging so there is one canonical repo.

---

## Step 1 — Commit and push the Dockerfile fix

The live service is currently held together by two Railway environment
variables (`HOME=/tmp`, `XDG_CACHE_HOME=/tmp/.cache`) that work around a real
bug in the image. The bug:

- The Dockerfile warmed the Chroma index **as root**, so chromadb's ONNX MiniLM
  model cached into `/root/.cache`.
- It then created the runtime user with `--no-create-home` and switched to uid
  10001, which can read neither `/root/.cache` nor a non-existent
  `/home/attest`.
- Result: `POST /v1/demo/query` returned `502` with
  `[Errno 13] Permission denied: '/home/attest'` on **every** call. The
  container booted fine and `/v1/health` returned `200`, so nothing caught it
  until a real query ran.

The fix is already in your working tree: the user is created with a home
directory *before* the warm step, `HOME` and `XDG_CACHE_HOME` are exported, and
the warm runs as uid 10001 so the model is baked in at a path that user owns.

```bash
git add Dockerfile dashboard/attest-dashboard.html dashboard/vercel.json DEPLOY.md
git commit -m "Fix container HOME for Chroma model cache; add dashboard auth + Vercel config"
git push frontier main
```

Railway auto-deploys on push. Watch it, then **remove the workaround** so the
image is the thing that is correct rather than the environment:

Railway → `attest-api` → Variables → delete `HOME` and `XDG_CACHE_HOME` → redeploy.

Verify before moving on:

```bash
curl https://attest-api-production.up.railway.app/v1/health
# {"ok":true,"version":"0.1.0"}
```

---

## Step 2 — Confirm Railway backend settings

Railway → `attest` → `attest-api`.

**Settings that must not change:**

- **Replicas: 1.** The SSE event bus in `attest/api/stream.py` is process
  memory. Two replicas do not error — they silently split the bus, and a
  dashboard connected to replica B never sees events published on replica A.
  Same reason `--workers` is absent from the Dockerfile `CMD`.
- **Health check path:** `/v1/health` (from `railway.json`). It is
  unauthenticated by design; it returns only `{"ok", "version"}`.

**Environment variables currently set:**

| Variable | Value | Note |
|---|---|---|
| `SUPABASE_URL` | `https://fcnepzhildbupbgqalqk.supabase.co` | |
| `SUPABASE_KEY` | service_role JWT | **must** be service_role — RLS is forced, the anon key gets nothing |
| `GROQ_API_KEY` | set | effectively the primary provider |
| `GEMINI_API_KEY` | set | fallback |
| `TAVILY_API_KEY` | set | independent web retrieval |
| `ATTEST_API_KEY` | set | bearer token for every route but `/v1/health` |
| `APP_ENV` | `prod` | enables HSTS |
| `TRUSTED_PROXY_HOPS` | `1` | correct for Railway's single edge proxy; without it every caller shares one rate-limit bucket |
| `MAX_BUDGET_USD` | `0.25` | server-side clamp on caller-supplied `budget_usd` |
| `CORS_ALLOW_ORIGINS` | `*` | **tighten in step 4** |
| `LOG_LEVEL`, `APP_VERSION` | `INFO`, `0.1.0` | |

**`ANTHROPIC_API_KEY` is not set** because it is empty in your `.env`. Add it in
Railway if you get a key — see the warning at the end of this document.

---

## Step 3 — Deploy the dashboard to Vercel

The dashboard is one static HTML file. No build step, no framework.

`dashboard/vercel.json` is already committed and rewrites `/` to
`/attest-dashboard.html`, so the site works at the bare domain.

1. Go to <https://vercel.com/new> and sign in with GitHub.
2. **Import** `Pragatish616/ATTEST_FRONTIER`. Grant repo access if prompted.
3. Configure the project:
   - **Framework Preset:** `Other`
   - **Root Directory:** `dashboard`  ← click *Edit* and select the folder
   - **Build Command:** leave empty (toggle the override off)
   - **Output Directory:** leave empty
   - **Install Command:** leave empty
4. Do **not** add any environment variables. The API key is typed into the page
   at runtime; it must never be baked into a static asset that anyone can view
   the source of.
5. **Deploy.** You get something like
   `https://attest-frontier.vercel.app`.
6. Open it. The API field is pre-filled with the Railway URL. Paste your
   `ATTEST_API_KEY` into the key field and run a query.

If you want a nicer hostname: Vercel → Project → Settings → Domains → rename
the `.vercel.app` subdomain.

---

## Step 4 — Lock CORS to the Vercel origin

Only now does the origin exist. In Railway → `attest-api` → Variables:

```
CORS_ALLOW_ORIGINS = https://<your-project>.vercel.app
```

Comma-separate if you add a custom domain or want preview deploys:

```
CORS_ALLOW_ORIGINS = https://attest-frontier.vercel.app,https://attest.yourdomain.com
```

Two consequences, both deliberate:

- `attest/api/main.py` sets `allow_credentials` to `True` the moment the origin
  list is not a wildcard. The dashboard sends a bearer header, not cookies, so
  this changes nothing for it — but a wildcard combined with
  `Allow-Credentials: true` is spec-invalid and browsers reject it, which is
  exactly why the code ties the two together.
- **Opening `dashboard/attest-dashboard.html` from your local filesystem will
  stop working**, because `file://` sends `Origin: null`. For local testing
  after this step, serve it over HTTP and add that origin too:

  ```bash
  cd dashboard && python -m http.server 5500
  # then add http://localhost:5500 to CORS_ALLOW_ORIGINS
  ```

Redeploy the Railway service after changing the variable.

---

## Step 5 — Verify the deployed stack

```bash
API=https://attest-api-production.up.railway.app
KEY=<your ATTEST_API_KEY>

# 1. health — no auth
curl $API/v1/health

# 2. auth is actually enforced — expect 401
curl -i $API/v1/runs

# 3. authenticated read — expect 200
curl $API/v1/runs -H "Authorization: Bearer $KEY"

# 4. full pipeline — expect 202 + run_id + a generated answer
curl -X POST $API/v1/demo/query \
  -H "Authorization: Bearer $KEY" \
  -H "content-type: application/json" \
  -d '{"query":"How long do I have to return an opened product?","k":4}'

# 5. the trace — claims, per-verifier verdicts, probes
curl $API/v1/runs/<run_id> -H "Authorization: Bearer $KEY"
```

Then open the Vercel URL and run the same query through the UI. Steps 1–5 have
all been run against this deployment and passed.

---

## Rollback

- **Bad backend deploy:** Railway → `attest-api` → Deployments → pick the last
  green one → *Redeploy*.
- **Bad frontend deploy:** Vercel → Deployments → previous → *Promote to
  Production*.
- **Schema:** `migrations/001_init_rollback.sql` and
  `migrations/002_rls_rollback.sql`. Note that rolling back `002` re-opens every
  trace table to the anon key.

---

## Known gaps in what is deployed

Be able to say these out loud before a judge finds them.

1. **`ANTHROPIC_API_KEY` is empty, so the stack runs on Groq
   `llama-3.1-8b-instant`.** Your README and PLAN both name Anthropic as the
   primary provider. The decomposer visibly degrades on the smaller model — it
   has split sentence fragments (`"If you install SupportBot Desktop on Windows
   10,"`) into standalone "atomic claims". Fix this first if you fix anything.

2. **FRAGILE has never fired.** Across every run in the database, 36 of 57
   probes failed to flip when they were expected to — the prober is working —
   but `FRAGILE` has never survived reconciliation. The chain: FRAGILE needs a
   `GROUNDED` entailment baseline; the corpus is fictional so the independent
   verifier finds no web evidence and returns `UNSUPPORTED`; and precedence puts
   `UNSUPPORTED` above `FRAGILE`. Your headline differentiator is structurally
   invisible. The candidate fix — have the independent verifier return
   `UNVERIFIABLE` rather than `UNSUPPORTED` when it finds no evidence at all —
   touches no frozen contract.

3. **STALE fires rarely and did not fire on its seeded case.** "Windows 10
   continues to receive security patches from Microsoft" came back `GROUNDED`
   from all three verifiers including the independent one.

4. **Benchmark numbers are still `PENDING`.** `bench/results.md` has no real
   run behind it. This is the largest gap against PLAN.md §13.

5. **Single replica only.** In-memory SSE bus. Documented, not fixed.

6. **Rate limiting is in process memory.** Correct only because there is exactly
   one replica. If that changes, this moves to Redis at the same time.

7. **`pg_net` is installed in Supabase.** It was added to smoke-test the
   deployed API from SQL. Harmless, but drop it if you want a clean database:
   `drop extension pg_net;`
