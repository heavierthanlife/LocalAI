---
name: qa-loop
description: Run the LocalAI incremental QA loop — /qa-loop, "跑一轮 qa-loop", "QA loop", 增量评审 last_head..HEAD, "消费 pending.flag", Round NNN. Reviews unreviewed commits with read-only mimo-v2.6-pro reviewers plus mandatory cross-review, triages Critical/High/Medium/Low, fixes in approved batches with deepseek-flash writers, verifies gates, documents, pushes, rebuilds the image, and re-checks. Use when asked to review the delta since data/qa_loop/last_head or to continue/close the QA loop.
---

# LocalAI QA-Loop

Incremental, auditable review loop over `data/qa_loop/last_head..HEAD`. Findings are
triaged **C/H/M/L**, fixed only in **user-approved batches**, then verified, documented,
pushed, and re-checked against the rebuilt image.

**Stop gate**: this round adds **0 Critical and 0 High** AND `pending.flag` is empty.

## When to Use

Use for: `/qa-loop` · "跑一轮 qa-loop" · "QA loop" · "增量评审 last_head..HEAD" ·
"pending.flag 待运行" · "Round NNN" · "继续上一轮 QA".

Do **not** use for: a single-file review, a bug report with a known cause, or anything
where no `last_head` baseline exists. This is the audit loop, not general code review.

## Non-negotiable Rules

1. **Manual trigger only.** Never run from `post-commit` or any hook. The hook's sole job
   is to write `data/qa_loop/pending.flag` (anti-self-trigger).
2. **Per-batch approval before any edit.** Present the ④ CONFIRM batch plan and wait.
   Do not start IMPLEMENT on inferred consent.
3. **Exactly one writer at a time.** Reviewers are strictly read-only.
4. **Never advance `last_head` until ⑨ RE-CHECK passes.** Advancing early destroys the
   audit baseline.
5. **No silent mode switch.** If subagent launch, model, or tooling fails, stop and report
   the exact failure. Do not fall back to another execution mode on your own.

## Roles, Models, Tools

Declared in `.pi/qa-loop.project.md`; the table below is the fallback. Resolve every path
from the environment — **never hardcode a user name, HOME, WSL mount, or drive letter.**

| Role | Agent | Model | Tools | Notes |
| --- | --- | --- | --- | --- |
| Reviewer (read-only) | `reviewer` | `xiaomi-token-plan-cn/mimo-v2.6-pro` | `read,grep,find,ls` | Unlimited; parallelise freely |
| Cross-examiner (read-only) | `reviewer` | `xiaomi-token-plan-cn/mimo-v2.6-pro` | `read,grep,find,ls` | Fresh context, adversarial |
| Writer | `worker` | `deepseek/deepseek-flash` | full incl. `edit,write` | Swappable; one at a time |
| Visual subagent | `reviewer` | `xiaomi-token-plan-cn/mimo-v2.6-pro` | `read` (+ image) | **Mandatory cross-check** |

Reviewers have **no `bash`** — they cannot run `git diff`. Supply the diff as files (below).

## Review Shard Protocol (do this before launching reviewers)

```bash
R=$(cat data/qa_loop/last_head)..HEAD
mkdir -p .remember/tmp/rNNN                     # .remember/ is gitignored — safe scratch
git diff $R -- <group-A paths> > .remember/tmp/rNNN/a-backend.diff
git diff $R -- <group-B paths> > .remember/tmp/rNNN/b-frontend.diff
git diff $R -- <group-C paths> > .remember/tmp/rNNN/c-records.diff
wc -c .remember/tmp/rNNN/*.diff                  # each MUST be < 50000 bytes
```

`read` truncates at 2000 lines or 50 KB, so no shard may exceed 50 KB. Then **prove
coverage and parity** — a shard set that silently drops a file is the worst failure mode:

```bash
git diff --name-only $R | sort > .remember/tmp/rNNN/_all.txt
git diff --name-only $R -- <all grouped paths> | sort > .remember/tmp/rNNN/_cov.txt
comm -23 .remember/tmp/rNNN/_all.txt .remember/tmp/rNNN/_cov.txt   # MUST be empty
cat .remember/tmp/rNNN/*.diff | wc -c
git diff $R | wc -c                                                # MUST be equal
```

Record the shard list, byte sizes, and the empty `comm` output in the round file's ①.

## Procedure

### ① COLLECT

1. Read, in order: `.pi/qa-loop.project.md` → `data/qa_loop/last_head` →
   `data/qa_loop/pending.flag` → `data/qa_loop/config.json` (`max_rounds`, default **3**)
   → the last `round-NNN.md` (template) → `data/unresolved.yaml` (pending items to fold in)
   → `data/fix_registry.yaml` + `tests/test_regression.py` (invariants).
2. Record the delta: `git rev-list --reverse $R` and `git diff --name-status $R`.
   Separate **code-bearing** from docs-only files.
3. Confirm `HEAD` vs `LocalAI/master`: `git rev-list --left-right --count master...LocalAI/master`
   must be `0 0`. Worktree must be clean.
4. Build the shards and run the coverage/parity proof above.
5. Launch **3 independent read-only reviewers in parallel** (same shard set, three
   different angles — this is the first half of cross-review):
   - **R1 — correctness & security.** Injection, path traversal, auth/ownership
     fail-open vs fail-closed, unhandled exceptions on real call paths, resource leaks,
     supply-chain/build correctness.
   - **R2 — frontend behaviour & render paths.** Every path that renders server data
     (`innerHTML`, `href`/`src` assignment, `eval`, template strings), pre-login loaders,
     empty states, console errors, event-listener/index bugs, mobile/narrow layout.
   - **R3 — invariants, mirrors & test adequacy.** Cross-file mirror pairs, registry
     completeness, whether the added tests would actually catch the regression they claim.
6. Every finding MUST carry `file:line`, the concrete trigger, and the observable
   consequence. Demand it; reject findings without a line number.
7. Optional ⑤ visual shard (below) when UI changed.

### ② VERIFY (mandatory — never trust a finding as-is)

For **every** finding, open the cited line at `HEAD` and decide 有效 / 误报 / 降级,
writing the evidence. A reviewer's output is a hypothesis, not a verdict. Findings the
main agent cannot reproduce against `HEAD` are downgraded or dismissed explicitly.

### ③ TRIAGE + CROSS-REVIEW

**Cross-review is mandatory, not optional.** Two layers:

1. **Cross-examination pass.** Launch a **fresh** `mimo-v2.6-pro` reviewer with all three
   finding lists plus the shards. It must:
   - attempt to **refute every Critical and High** with counter-evidence, and say so when
     it cannot;
   - surface findings **only one** reviewer caught (blind spots);
   - flag duplicates, contradictions, and severity inflation.
2. **Dispute round.** Any C/H where reviewers disagree goes to a separate independent
   reviewer with both positions stated neutrally. The main agent adjudicates and records
   which position won and why. Unresolved disputes are **not** auto-fixed — report them.

**Adversarial-seat constraints** (learned the hard way, Round 031 — three failures with the
same model as the reviewers):

- **Prefer a different model family for the adversarial seat.** A same-model cross-examiner
  shares the reviewers' blind spots and cannot refute them. Independence is the whole point.
- **Keep the task narrow.** Pass the finding lists + shard paths, ask only for: refute each
  C/H, name blind spots, dedupe, rule on disputes, and the adjudicated table. Never ask a
  child to hand-count or exhaustively cross-reference a multi-thousand-line file — doing so
  made `mimo-v2.6-pro` degenerate into incoherent repetition (42 message starts, 2
auto-retries, no coherent output) on `tests/test_regression.py`.
- **Feed large inputs as paths, never inline.** An inline ~25 KB brief stalled the child;
  passing file paths and letting the child `read` them works.
- **Fail visibly.** If the adversarial pass fails twice, **stop and report** the exact
  failure. A decorative gate that silently passed is worse than a known gap. Record which
  findings rest on main-agent adjudication alone, and say so in §③.

Then assign final severity (C/H/M/L) and group into batches:
**A** backend correctness · **B** security · **C** frontend/functional · **D** hardening.

### ④ CONFIRM — stop and wait for approval

Write the batch plan into `data/qa_loop/round-NNN.md` §④ as a checkbox list:
`[batch] | severity | file:line | one-line fix`. List rejected/downgraded items with the
reason. **Wait for the user.** This is the loop's approval gate.

### ⑤ IMPLEMENT

Only `deepseek-flash` (`worker`), one writer at a time, batch by batch. Per fix:

- code change (repo style: no comments unless necessary; `to_rel_path()`/`resolve_path()`,
  never hardcoded absolute paths);
- a `data/fix_registry.yaml` entry recording the invariant that must hold — use
  `type: literal` when the pattern contains regex metacharacters;
- a regression test in `tests/test_regression.py`.

Commit per batch with `type: 中文摘要`.

### ⑥ VERIFY (independent)

```bash
.venv/Scripts/python.exe scripts/verify_fixes.py
set PYTEST_ADDOPTS=-p no:capture
.venv/Scripts/python.exe -m pytest tests/test_regression.py -q
.venv/Scripts/python.exe -m pytest tests/test_smoke.py -q
.venv/Scripts/python.exe scripts/check_doc_drift.py
.venv/Scripts/python.exe scripts/check_system.py
node --check static/js/<each changed>.js
```

Paste the **actual output** — never claim a gate passed without it. If the change touches
`app/services/`, `app/routes/`, `celery_app.py`, `data/fix_registry.yaml`, or the
compliance/clearance path, also run the 3/3 baseline regression (工程/货物/服务) and put
`regression: N/3 baseline passed` in the commit message.

### ⑦ DOCS

`CHANGELOG.md` top entry split by FIX id (`FIX-<date>-QA-<id>`) · `AGENTS.md` /
`docs/MANIFEST.md` if conventions or layout changed · regenerate
`repair_kit/SYSTEM_CHECKLIST.md` via `check_system.py` · fill in the round file §⑤–⑨.
Docs ship in the same commit as the code.

### ⑧ PUSH

`git push LocalAI master`, then assert the worktree is clean and
`git rev-list --left-right --count master...LocalAI/master` is `0 0`.

### ⑨ IMAGE (only when `.pi/qa-loop.project.md` sets `has_docker: true`)

```bash
docker compose build
docker compose up -d --force-recreate app celery-worker celery-beat   # leave nginx/pg/redis
curl -k -s -o /dev/null -w "%{http_code}" http://localhost:8000/check_auth   # expect 200
```

Then prove **image == HEAD** with the in-container grep spot-checks listed in
`.pi/qa-loop.project.md`. Any miss = deployment failure: roll back the previous image and
stop the round.

### ⑩ RE-CHECK / STOP GATE

Launch a **fresh** read-only reviewer against the reworked diff only. Count the newly
introduced C/H.

- **new C == 0 AND new H == 0 AND `pending.flag` empty** → stop gate met, loop ends.
  Now write `HEAD` into `data/qa_loop/last_head` and clear `pending.flag`.
- Otherwise, if rounds run < `max_rounds`, keep the baseline and start the next round;
  else stop and report the remaining items into `data/unresolved.yaml`.

## Visual Shard (only when UI changed)

Goal: get the **most accurate** reading, and never let one model's guess drive a fix.
**The crop method is the dominant error source — not the reader.** Budget effort there.

### Step 1 — capture crops that provably contain their text

Never use full-page captures (a tall page previously came back ~98% blank). Never trust
`getBoundingClientRect()`: an element's box can miss its own glyphs entirely. And
`Range.getClientRects()` still reports text that an ancestor clipped or a modal occluded.
Build crops with this ladder, and discard any crop that fails it:

1. **Candidate text** = text nodes not inside an icon-font subtree. `Material Symbols`
   ligature names (`build`, `key`, `edit_note`, `local_fire_department`) appear in
   `innerText` but are NOT visible text — they poison ground truth. Strip them.
2. **Crop box** = union of those nodes' **clipped** rects, intersecting each rect with the
   client rect of every ancestor whose overflow is `hidden|clip|auto|scroll`.
3. **Visibility proof (occlusion).** Sample a grid (~3 px) **inside each node's own rect**
   and require `document.caretPositionFromPoint(x, y)` to return that same node for ≥30%
   of samples. A sample returning a different node means an occluder is on top → drop the
   node. `caretPositionFromPoint` alone is too permissive — it snaps to the nearest caret
   even outside glyph ink, so sampling an edge still returns the run and over-claims truth.
4. **Pixel gate.** Reject near-uniform crops (`stddev < 12`, or ink `< 1.5%` of pixels).
   This gate alone is NOT sufficient: an occluding modal's own ink can make a
   blank-for-its-target crop look valid. The needle must also appear in the hit-tested text.
5. **Sanity.** Characters must fit the box (compare `width / chars_per_line` with the font
   size). 21 characters claimed inside 79×31 px means the truth is wrong, not the reader.

Record each crop's box, ink %, and hit-tested text in the round file.

### Step 2 — A/B both readers on the same crops

`mimo-v2.6-pro` vision vs `app/services/ocr.OCRManager` (EasyOCR), scored on character
accuracy against the hit-tested DOM text (whitespace-stripped).

Measured on the pre-login gate UI (1440×900 viewport, Simplified Chinese, 27–31 px crops):

| Reader | mean char-acc | exact |
| --- | --- | --- |
| EasyOCR (CPU, `ch_sim+en`) | **0.908** | 2/5 |
| `mimo-v2.6-pro` vision | see round file | — |

EasyOCR's residual errors were systematic: `已`→`2`, `已`→`己`, `PIN`→`pN`, full-width
`（）`→half-width `()`, `AI`→`A`. It returns nothing or garbage when a crop is downsampled
or its text is small relative to the frame — which is exactly why the crop method decides
the score.

### Step 3 — cross-review for vision is mandatory

EasyOCR is unreliable on small UI text and returns nothing on downsampled pages;
`mimo-v2.6-pro` is strong on UI text and layout but has demonstrated **false positives and
false negatives** (it once reported a missing glyph that DOM proved was loaded, and missed
present emoji). Therefore: run **two blind `mimo-v2.6-pro` passes** over the same crops in
different orders, then one reconciliation pass that must list every disagreement, judge
which pass is right, and flag suspiciously fluent text as possible hallucination. Any mimo
claim that would drive a fix must additionally be confirmed by DOM. Material disagreement →
mark the finding 「截图方法待复核」 and do not fix on it.

### Step 4 — record

Put the A/B table (per reader: correct / wrong / missed), both VL passes and the
reconciliation outcome in the round file §V, and state which reader you trust for which
question.

## TDI Is Advisory Only — Never a Gate

`/lens-tdi` reads the accumulated `metrics-history.json`, which has real defects:

- the sample is only the files pi-lens happened to analyse (one run scored **2 files**
  while the review graph knew **182**);
- **deleted files are never pruned**, so a deleted scratch script kept contributing;
- pi-lens indexes `.git/` scratch files (`.git/resolve_conflicts.py`, `.git/*_msg.txt`);
- MI is dominated by `16.2·ln(LOC)`, so it mostly measures file **length** — the mandated,
  monotonically growing `tests/test_regression.py` looks bad by construction;
- entropy only strips `//` and `/* */`, so Python `#` comments are counted as "code
  unpredictability".

Use TDI for trends only. Never gate, block, or prioritise work on it. If asked for a real
number, run a full-workspace scan first (`lens_diagnostics` with `scope=workspace`,
`refreshRunners=all`) and still report the caveats.

## Pitfalls

- **No `bash` on this machine.** Git Bash is absent. Use `cmd`/PowerShell, or drive
  commands through `.venv\Scripts\python.exe`. Do not assume `bash -lc` works.
- `pytest` needs `PYTEST_ADDOPTS=-p no:capture` (Windows Python 3.12 teardown
  `ValueError: I/O operation on closed file`).
- `.venv` has **no `pip`** — `.venv/Scripts/python.exe -m ensurepip` first.
- Proxy `http://127.0.0.1:2099` is required for git/GitHub and npm.
- Docker apt needs **ustc https** (tuna 403; aliyun http 404 on trixie).
- `data/` assets are absent inside containers (`.dockerignore` + the `app_data` volume);
  mount `:ro` or seed via `app/bootstrap.py::ensure_seeded()`.
- `read` truncates at 50 KB — undersized shards, not one big diff.
- Reviewers cannot run `git diff`; feed them files.
- Don't hardcode the reviewer launcher, HOME, or a WSL mount.
- Commit messages: `type: 中文摘要`. Never commit keys/private keys/binaries
  (`cert/key.pem`, `msedgedriver.exe`).
- `/qa-loop` is never automated — `.pi/qa-loop.project.md` and AGENTS.md both require it.

## Verification

This round is complete only when all of these hold:

- `data/qa_loop/round-NNN.md` exists with §①–⑨ filled, including the shard byte sizes,
  the empty `comm` coverage output, the A/B visual table (if UI changed), and the
  cross-examination outcome.
- Every ⑥ gate was shown with its real output; backend/security changes carry a fresh
  `data/fix_registry.yaml` entry and a `tests/test_regression.py` case.
- `git rev-list --left-right --count master...LocalAI/master` is `0 0` and the tree is clean.
- If `has_docker`, every in-container spot-check matched (image == HEAD).
- `last_head` advanced **only** after ⑨ RE-CHECK, and `pending.flag` is cleared.
- Round file, code, registry, and docs are in the same commit history as the fixes.
