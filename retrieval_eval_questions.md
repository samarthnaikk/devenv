# Retrieval Evaluation — 16 Questions

Generated from the local Codex (`~/.codex/sessions/**/*.jsonl`, `session_index.jsonl`) and
OpenCode (`~/.local/share/opencode/opencode.db`) archives.

Projects covered: `devenv`, `facepred`, `get-drip`, `hirex-frontend`, `DigiX`, `vaxo1a`,
`temp1` (C++ ARQ), `latexee`, `cd1` (C semaphores).

## How to use this file

- Ask each question from inside the retrieval workspace and capture which prior sessions
  were recalled.
- A question counts as **retrieved** only if the ground-truth session listed in the answer
  key appears in the recalled set *and* the agent's answer contains the expected fact.
- Part A (Q1–Q7) is deliberately **niche**: each answer hinges on an exact identifier,
  value, or decision that exists in only one session. These are the retrieval stress tests.
- Part B (Q8–Q16) is **follow-up style**: "we hit this error on this project, how did we fix
  it?" Each has one primary source session.
- Q16 was added from a real runtime miss: the correct session was retrieved but its answer
  never reached the model's context, so evaluate both session recall and answer coverage.
- Before evaluating, re-run `scripts/backfill_session_embeddings.py` so the vectors cover the
  latest archives. Note: recent meta sessions (including review/exploration sessions) can
  themselves mention some of these facts, so score against the ground-truth session id, not
  just "a session that mentions the keyword".

---

## Part A — Niche questions (7)

**Q1.** In `devenv`, how does it actually invoke OpenCode for a live model call, and where
does it read OpenCode session history from?

**Q2.** In the `facepred` pair-prediction output, what produces `similarity_score` and
`match_confidence`, and what exact threshold value decides `same_person`?

**Q3.** What is the `CORAL` head in `facepred`'s recommended age model, and how is the
MiVOLO backbone called (inputs and resolution)?

**Q4.** For whole-session embeddings in `devenv`, what table and key format are used, and how
many Codex and OpenCode sessions were indexed in that run?

**Q5.** In `get-drip`, what exact command runs the disposable persona test for Vineeth S at
Agora, and what external error did the first run hit?

**Q6.** What are the exact Celery queue names `DigiX` partitions async work across?

**Q7.** Why did the `vaxo1a` JEE markdown parser only extract 57 of 75 questions from the
first paper, and what were the regex fixes?

---

## Part B — Follow-up questions (9)

**Q8.** On `devenv`, the web runtime crashed with a `SyntaxError` in `context_builder.py` —
what was the malformed line and how did we fix it?

**Q9.** On `devenv`, we hit `Uncaught ReferenceError: pendingRunMode is not defined` in the
website Composer — what was the root cause and the fix?

**Q10.** On `get-drip`, why did `workspaceInviteEvents` fail Convex schema validation under
`bun run convex:dev`, and how did we fix it?

**Q11.** On `get-drip`, what caused the esbuild `duplicate-object-key` warnings during Convex
bundling, and where were the duplicates?

**Q12.** On `hirex-frontend`, the backend crashed on startup with a `JSONB` error after the
Qualifly track migration — what was the error and how did we fix it?

**Q13.** In `temp1`'s `selective_repeat.cpp`, why did `./a.out` hang after entering the lost
frames, and how did we fix it?

**Q14.** On the `latexee` Hugging Face Space, the build looked like it failed — was
`pdflatex` missing, and what were the two real problems?

**Q15.** On the `cd1` semaphore programs, why did they compile but print wrong values on
macOS, and what did we switch to?

**Q16.** On `hirex-frontend`, why did the recruiter dashboard go blank right after saving
details, and where exactly was the bug?

---
---

# Answer key (ground truth)

Do not feed this section to the agent under test.

## Q1 — devenv OpenCode invocation and history source

- **Project:** devenv
- **Source:** Codex `019f3295-8775-7b11-9373-7ae47cc6158b` — "Locate opencode connection"
- **Expected answer:** Devenv does not use an OpenCode SDK/HTTP API for live calls; it shells
  out to the local CLI: `opencode run --format json --dir <workspace> [--model <model>] <prompt>`.
  Parsed by `OpenCodeAICore` in `core/ai/routing.py` (gated by `RoutingAICore`). Session
  history is read from the SQLite DB `~/.local/share/opencode/opencode.db` via
  `OpenCodeSessionProvider` in `core/runtime/context_builder.py`.
- **Proof:** `` `opencode run --format json --dir <workspace> [--model <model>] <prompt>` ``

## Q2 — facepred prediction field provenance

- **Project:** facepred
- **Source:** Codex `019ed3cf-2c69-7142-863b-45e37827f0f8` — "Identify model parameters"
- **Expected answer:** `raw_cosine_score` = antelopev2 embedding cosine;
  `similarity_score = (raw_cosine_score + 1) / 2`; `pipeline_match = similarity_score >= 0.75`;
  `match_confidence` comes from a `sklearn.pipeline.Pipeline(... LogisticRegression ...)` saved
  at `artifacts/models/fast_model.joblib`; `same_person` is `match_confidence >= match_threshold`
  where `match_threshold` is the constant `FAST_DECISION_THRESHOLD = 0.48`. Ages come from the
  MiVOLO + CORAL path, and `predicted_bias` from `artifacts/rectifier_model.pkl`
  (`poly_coeffs`, `rf_model`, `shift_offset`).
- **Proof:** `It is the constant FAST_DECISION_THRESHOLD = 0.48.` and `"similarity_score": 0.864617,`

## Q3 — facepred CORAL head and MiVOLO call

- **Project:** facepred
- **Source:** OpenCode `ses_1529444f7ffesru6Hg6AZM9Gd9` — "MiVOLO pretrained vs best.pth model choice"
  (also `ses_1547ff430ffeo7cA8L0gM3BkFU` — "Best face prediction model choice")
- **Expected answer:** The recommended path uses the pretrained MiVOLO backbone
  `iitolstykh/mivolo_v2` via `AutoModelForImageClassification.from_pretrained(...)`, resized to
  `224x224`, fed `faces_input` plus a blank `body_input` (`[None]`) so only the face path is used.
  The `CoralHead` is `Linear -> ReLU -> Dropout -> Linear(1) + per-threshold bias`, trained with
  ordinal `levels` and `binary_cross_entropy_with_logits`. Saved as
  `artifacts/recommended_age_model/best_coral_age.pt`; the legacy scratch model
  `output_selfie_age/checkpoints/best.pth` is no longer recommended.
- **Proof:** `the model gets faces_input plus a blank body_input ([None]) so it uses the face path only.`

## Q4 — devenv unified session embeddings

- **Project:** devenv
- **Source:** Codex `01a01e0a-3788-7de1-adf6-246e779febc9` — "Store unified session embeddings"
- **Expected answer:** A persistent `external_session_embeddings` table in `memory.db`, keyed by
  `codex:<session_id>` / `opencode:<session_id>`, with content-hash dedupe. Provider roots come
  from `_default_provider_configs()` in `core/runtime/context_builder.py`. The indexing run
  reported `{"codex_session_files": 162, "opencode_sessions": 174}`.
- **Proof:** `{"codex_session_files": 162, "opencode_sessions": 174}`

## Q5 — get-drip persona test

- **Project:** get-drip
- **Source:** Codex `019eeef5-bc37-7e22-bf7c-b4e807d9d165` — "Add persona test script"
- **Expected answer:** Disposable script `scripts/test-persona.ts` calling the read-only debug
  action in `convex/context_building/persona/index.ts`; command:
  `bun scripts/test-persona.ts --full-name "Vineeth S" --company "Agora" --domain "agora.io" --email "vineeth.s@agora.io"`.
  First run failed because Bright Data returned HTTP 400 "Customer is not active"; once active it
  returned `persona: "developer"`, `confidence: "high"`, `score: 90`.
- **Proof:** `Bright Data LinkedIn profile request failed (400): Customer is not active`

## Q6 — DigiX Celery queues

- **Project:** DigiX
- **Source:** OpenCode `ses_1432beb90ffe00PaDLl4gh5zvj` — "New session - 2026-06-12T17:14:36.015Z"
  (parent of the `@explore subagent` session; subagent sessions are excluded as derived)
- **Expected answer:** Three Celery queues: `api_calls`, `web_scraping`, `heavy_compute`.
  Backend: FastAPI + Uvicorn + async SQLAlchemy + Alembic + Celery + Redis on PostgreSQL 15.
  Frontend: React 19 + Vite 8 + Tailwind v4.
- **Proof:** `Celery workers (3 queues: api_calls, web_scraping, heavy_compute)`

## Q7 — vaxo1a JEE parser bug

- **Project:** vaxo1a
- **Source:** OpenCode `ses_1a671836effe1pnxLSExwXKPw5` (project `vaxo1a`)
- **Expected answer:** On line 55 the answer key for Q4 and the Q5 marker run together
  (`...matQ5.`), so the `mat\s*$` check and the `\n`-prefixed Q-marker regex both miss it.
  Options rendered as raw LaTeX because `extractOptionsHtml()` returned plain text (skipping
  `wrapLatexText`) and the `\lt` guard matched `\left` as a substring. Fixes: always run
  `wrapLatexText`, replace the `\lt` guard with `\\(?:[a-zA-Z]{2,}|[()\[\]{}|])`, and change
  `extractAnswerKey` from `(\d+)` to `(-?\d+)` to handle `-560`. Final: paper 1 = 75/75,
  paper 2 = 71/75 (Q10–Q13 absent from source).
- **Proof:** `Look at line 55: ...matQ5. — Q4's answer key and Q5 marker run together on the same line.`

## Q8 — devenv context_builder SyntaxError

- **Project:** devenv
- **Source:** Codex `019f2859-910b-7c41-b468-55f5d172c7ed` — "Fix context_builder syntax error"
- **Expected answer:** Importing `from .context_builder import ContextBuilderService` raised
  `SyntaxError: invalid syntax` because line 1 of `core/runtime/context_builder.py` had been
  corrupted from `from __future__ import annotations` to `gfrom __future__ import annotations`.
  Fixed by restoring the line; the module then imported cleanly.
- **Proof:** `+gfrom __future__ import annotations`

## Q9 — devenv pendingRunMode ReferenceError

- **Project:** devenv
- **Source:** Codex `019f9f43-9c40-75b3-8b7f-64e83415b469` — "Fix pendingRunMode error"
- **Expected answer:** `Uncaught ReferenceError: pendingRunMode is not defined` at `Composer`
  (`app.js?v=bundle1:34272:116`). In `interface/website/src/components/Composer.js` the submit
  button render used `pendingRunMode`, but that identifier only existed inside `handleSubmit`.
  Fixed by defining a render-safe `const pendingRunMode = state.isRunning ? state.pendingRunMode : inferPendingRunMode({...})`,
  then rebuilding `interface/website/vendor/app.js` (`npm run build:vendor`) and running
  `npm run check:mount`.
- **Proof:** `the submit button render used pendingRunMode, but that variable only existed inside handleSubmit`

## Q10 — get-drip workspaceInviteEvents validation

- **Project:** get-drip
- **Source:** OpenCode `ses_1caea868fffeOc7bWKTOXY0bgn` — "Fix workspaceInviteEvents schema validation"
- **Expected answer:** `bun run convex:dev` rejected an existing document
  (`kx74tz0x1wbdxd6zd7e1fby8ax86tm5d`) whose `action` was `"sent"`, but the validator only
  allowed `created/resent/accepted/declined/expired`. Fix: add `v.literal("sent")` to the
  `action` union in `convex/schema.ts`. `"sent"` was a legacy value; the code path writes
  `"resent"`.
- **Proof:** `Path: .action Value: "sent" Validator: v.union(v.literal("created"), v.literal("resent"), ...)`

## Q11 — get-drip duplicate object keys

- **Project:** get-drip
- **Source:** Codex `019e8c3b-293e-7bd3-aa91-ccfec1dd15c7` — "Fix duplicate object keys"
- **Expected answer:** esbuild `duplicate-object-key` warnings: `displayName` duplicated at
  `convex/schema.ts:69` (original `:66`); in `convex/onboarding/workspace/mutations.ts`,
  `displayName` at `:621` (original `:618`), plus `timezone` (`:656`/`:654`) and `settings`
  (`:657`/`:655`) duplicated in the `updateUserProfile` update object. Duplicates removed
  without behavior change.
- **Proof:** `Duplicate key "displayName" in object literal [duplicate-object-key] ... convex/schema.ts:69:2`

## Q12 — hirex-frontend JSONB crash

- **Project:** hirex-frontend
- **Source:** OpenCode `ses_1cb77a611ffeyh4uz0oyOC5sH9` — "Add Design and Management evaluation tracks"
- **Expected answer:** Migration `backend/migrations/versions/0011_add_qualifly_track.py` adds
  `jobs.qualifly_track` and `qualifly_applications.external_links`. Backend startup crashed
  because `external_links` was declared `sa.JSONB()`; fixed by changing it to `sa.JSON()`.
  Separately, the AI interview threw `cannot access local variable 'user_turn_count'` because it
  was used before assignment — fixed by computing it earlier.
- **Proof:** `Changed sa.JSONB() to sa.JSON() - SQLAlchemy uses JSON for JSON columns in PostgreSQL.`

## Q13 — temp1 selective_repeat hang

- **Project:** temp1 (C++)
- **Source:** Codex `01a04442-9ea3-7730-bea3-84fec496bc39` — "Fix frame retransmission code"
- **Expected answer:** Lost frames were read with `while (cin >> x) lost.insert(x);`, which blocks
  waiting for more input until EOF, so `./a.out` appeared to hang. Fixed by flushing the line with
  `cin.ignore(numeric_limits<streamsize>::max(), '\n');` and reading with `getline` + `stringstream`
  (adding `<limits>`, `<sstream>`, `<string>`). A nested loop that printed bogus repeated
  `Frame 1 -> Delivered from buffer` lines was also fixed.
- **Proof:** `Your earlier version used: while (cin >> x) lost.insert(x); That keeps waiting for more input until EOF, which is why ./a.out looked like it was hanging.`

## Q14 — latexee Space build failure

- **Project:** latexee (Hugging Face Space)
- **Source:** Codex `019f0314-9290-74f0-a02e-e3ba995e68fa` — "Add LaTeX PDF rendering"
- **Expected answer:** `pdflatex` was **not** missing (the log shows it running:
  `pdflatex -interaction=nonstopmode ... "verticalaxis_0209.tex"`). The real LaTeX problem was
  `LaTeX Warning: File 'watermarklogo.jpg' not found on input line 156`, because `app.py` only
  accepted a single `.tex` input; fixed by accepting supporting attachments and copying them into
  the working dir before `latexmk`. The startup crash was
  `ModuleNotFoundError: No module named 'requests'`, fixed by adding `requests==2.32.3` to
  `requirements.txt`.
- **Proof:** `ModuleNotFoundError: No module named 'requests'` and `Latex Warning: File 'watermarklogo.jpg' not found`

## Q16 — hirex-frontend blank dashboard after saving

- **Project:** hirex-frontend
- **Source:** OpenCode `ses_14d2cb8e4ffeObVH00TuYhgs3w` — "Blank dashboard after saving details debugging"
  (parent of `ses_14d2c04eeffeBTFi0G16ZVC4lA` "Review blank screen (@explore subagent)")
- **Expected answer:** A React hooks-order bug in
  `frontend/src/pages/RecruiterDashboard/RecruiterDashboard.tsx`: `useMemo` was declared after the
  early returns for `showSettings`/`initialLoading`, so the first render returned early and the next
  render executed the hook, throwing "React has detected a change in the order of Hooks" /
  "Rendered more hooks than during the previous render". Fixed by moving the derived dashboard logic
  (including `useMemo`) above the conditional returns.
- **Proof:** `useMemo was declared after early returns for showSettings and initialLoading`


## Q15 — cd1 sem_init on macOS

- **Project:** cd1
- **Source:** Codex `01a06d43-2d85-73a3-8313-55a8057828ab` — "Find semaphore questions and add 3"
- **Expected answer:** The programs compiled but could print wrong values because `sem_init`
  (unnamed POSIX semaphores) is unsupported on macOS. Fix: rewrite the three semaphore programs to
  named POSIX semaphores using `sem_open`/`sem_wait`/`sem_post`; monitor variants keep
  `pthread_mutex_t` + condition variables. All six sources compile with
  `-Wall -Wextra -Werror -pthread`.
- **Proof:** `The local run exposed a real macOS issue: sem_init is unsupported there, so the programs could print incorrect values even though they compiled.`
