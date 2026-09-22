# Testing lab-agent by hand

Every command below is PowerShell, run from the project folder. Each one says
what it **proves** and whether it **costs money**. Only section 7 spends.

Run these three lines once per terminal session. Everything afterwards assumes
them:

```powershell
Set-Location -LiteralPath "C:\repos\ml_course_tasks\agent_frameworks\wwfinal_project\03_MVP_LabAgent"
$Py = ".\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"   # so em dashes and emoji print instead of "?"
```

---

## 1. Bring the lab up

```powershell
docker compose up -d
```
Proves: `lab-victim` (the OOM victim, 128 MiB limit) and `prometheus` are running.

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$((Resolve-Path $Py).Path)' -m agent.exporter"
```
Proves: the exporter is publishing on :9101. **Leave that window open** — it is a
host process, not a container, and Prometheus scrapes it. `-NoExit` keeps the
window open if it crashes so you can read why.

---

## 2. The observation path — free

```powershell
(Invoke-WebRequest -UseBasicParsing "http://localhost:9101/metrics").Content -split "`n" | Where-Object { $_ -match "^lab_container_" }
```
Proves: the exporter is answering, and with **our** metrics. A reply alone is not
enough — on Windows another program can hold the same port.

```powershell
(Invoke-RestMethod "http://localhost:9090/api/v1/targets").data.activeTargets | ForEach-Object { "$($_.labels.job) health=$($_.health) lastError='$($_.lastError)'" }
```
Proves: Prometheus is actually scraping. `health=up` with an empty `lastError`
is the line that matters.

```powershell
(Invoke-RestMethod "http://localhost:9090/api/v1/query?query=lab_container_restart_count").data.result | ForEach-Object { "$($_.metric.container) = $($_.value[1])" }
```
Proves: PromQL returns the restart count — the same API the agent's adapter uses.

```powershell
& $Py -m agent.wait_for_metrics
```
Proves: there are samples **fresh enough** for the adapter to accept (newer than
30 s). "Target up" and "data usable" are different facts.

---

## 3. Watch the incident happen — free

```powershell
foreach ($i in 1..10) { docker inspect lab-victim --format "t+$($i*3)s status={{.State.Status}} restarts={{.RestartCount}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}"; Start-Sleep -Seconds 3 }
```
Proves: `RestartCount` climbs while `Status` stays `running` — the container is
down for a fraction of a second. Note `exit=0 oom=false`: Docker **resets**
those on every start, which is why the adapter reads the daemon's event log.

```powershell
docker events --since 5m --until 1s --filter container=lab-victim --filter event=die --filter event=oom
```
Proves: where the truth lives — `oom` events and `die` with `exitCode=137`.

```powershell
docker logs --tail 20 lab-victim
```
Proves: nothing. An OOM-killed process cannot log its own death (SIGKILL cannot
be caught), which is why RB-002 sends you to `dmesg` instead.

---

## 4. The test suite — free, offline, no Docker needed

```powershell
& $Py -m pytest tests/ -q
```
Proves: 127 tests pass with no network, no API key and no containers. `MockLLM`
is the default in every one of them.

```powershell
& $Py -m pytest tests/ -q --ignore=tests/test_metrics_docker.py
```
Proves: nothing here depends on Docker Desktop at all — try it with Docker
stopped.

---

## 5. Safety: what the agent may NOT do — free

```powershell
& $Py -m pytest tests/test_registry.py -v -k "rejects"
```
Proves: the six refusals — a tool that is not on the allowlist, a write aimed at
`postgres`, names that only look like `lab-*`, a name that is not a string, and
argument names the tool does not have.

```powershell
& $Py -c "from tests.test_graph_mock import FORBIDDEN_REQUEST, run; from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM; final, llm, mem = run(llm=MockLLM([FORBIDDEN_REQUEST, DEFAULT_SCRIPT[1]])); print([l for l in final['history'] if 'REFUSED' in l][0])"
```
Proves: a model asking to restart `postgres` gets a REFUSED **observation** — the
run continues and the model is told why, instead of crashing.

```powershell
& $Py -c "from agent.tools.registry import ALLOWED; [print(tool, spec) for tool, spec in ALLOWED.items()]"
```
Proves: the policy table itself. This, not the prompt, is what grants.

---

## 6. Memory and retrieval — free

```powershell
& $Py -c "from agent.adapters.memory_sqlite import SqliteMemory; rows = SqliteMemory().recent_incidents('RB-002', hours=24); print('prior incidents:', len(rows)); [print(r['ts'], r['container'], '|', r['diagnosis']) for r in rows]"
```
Proves: what the agent remembers, in full. This survives every process.

```powershell
& $Py -m agent.corpus_experiment
```
Proves: the same retriever against three corpora. RB-002 alone (8 chunks), all
three runbooks (28 chunks, no dilution), and a generative-AI course (341 chunks,
confident nonsense). **This is the slide for the presentation.**

```powershell
& $Py -c "from agent.adapters.docs_tfidf import TfidfDocs; print(TfidfDocs().search('should I raise the memory limit?', k=1)[0])"
```
Proves: the runbook can answer the question the model kept getting wrong — the
DO NOT section, in full.

---

## 7. The agent itself — THIS SPENDS MONEY

Roughly 5,000–7,000 tokens per run (3 model calls, prompts of ~1,500 tokens).

```powershell
& $Py -m agent.wait_for_metrics
& $Py -m agent.run
```
Proves: the whole chain. Read the trace top to bottom:
- `DETECT` — Prometheus numbers, through `MetricsPort`
- `RECALL` — how many prior incidents, and how many runbook chunks
- `REASON` — what the model decided, and why
- `ACT` — what the dispatcher allowed it to do
- `RECORD` — what went into memory
- `VERDICT` — whether it cites RB-002 and the priors, and whether it still
  suggests raising the limit

```powershell
& $Py -m agent.probe
```
Proves: one single call, with the raw reply printed exactly as the model sent it
and then validated. Use this when a key or model name is wrong — it costs one
call instead of a whole run.

---

## 7b. The human gate and the audit trail — free

```powershell
& $Py -m agent.gate_demo
```
Proves: the same write approved once and denied once, in one process, with the
trail each answer leaves. No Docker, no model.

```powershell
& $Py -m agent.run --rehearse
```
Proves: a run that stops at the gate and **exits**. The request is printed; the
paused run is in `lab_checkpoints.db`. `--rehearse` scripts the MODEL only —
everything else is real — because since Stage 4 the real model usually declines
to restart, which is correct and makes the gate impossible to demo on demand.

```powershell
& $Py -m agent.run --rehearse
```
Proves: run it again and it refuses — one investigation per container at a time.
This process learned that from the file, not from the one that paused.

```powershell
& $Py -m agent.run --deny --rehearse
```
Proves: the denial reaches the model as an observation, and lands in the trail.
Nothing was executed.

```powershell
& $Py -m agent.run --approve --rehearse
```
Proves: the approval runs the tool — this really restarts `lab-victim`. Check
`RestartCount` before and after; a manual restart resets it to 0.

```powershell
& $Py -c "from agent import audit; [print(r['ts'][11:19], r['event'], r.get('approval') or r.get('answer') or '') for r in audit.read_all()[-8:]]"
```
Proves: one denial and one approval, both on disk, with who answered.

---

## 8. Shut down

```powershell
docker compose down
```
Stops `lab-victim` and `prometheus`. Close the exporter window by hand (Ctrl+C).

`lab_agent.db` survives on purpose — it is the agent's memory. Delete it if you
want the next run to start from "0 prior incidents":

```powershell
Remove-Item lab_agent.db
```

---

## If something fails

| Symptom | Cause | Fix |
|---|---|---|
| `MetricsUnavailable: no lab_container_* series` | the exporter window is closed | restart it (section 1) |
| `MetricsUnavailable: samples are N s old` | the exporter died, Prometheus still answers from its 5-minute lookback | restart the exporter |
| `LLMUnavailable: ... 401/403` | credentials | check `.env`; the course gateway returns 403 at Cloudflare Access |
| `LLMUnavailable: ... 400 temperature` | the model rejects non-default temperature | leave `LLM_TEMPERATURE` unset |
| target `health=down`, connection refused | nothing is listening on 9101 | restart the exporter |
| `A run on 'lab-victim' is already paused` | a gate is waiting for an answer | `--approve` or `--deny` it |
| `Nothing is waiting for an answer` | no paused run on that thread | start one: `python -m agent.run` |
| headings print as `Confirm ? read-only` | Python's stdout is on the Windows ANSI codepage | `$env:PYTHONIOENCODING = "utf-8"` |
