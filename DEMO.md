# The 3-minute demo

Six commands. The agent looks at a real broken container and diagnoses it with a
real model. Everything else this project can do is in `TESTING.md`.

**Cost:** one run is about 3 model calls and 5,000–7,000 tokens.

---

### 0. Open PowerShell and set up the session

```powershell
Set-Location -LiteralPath "C:\repos\ml_course_tasks\agent_frameworks\wwfinal_project\03_MVP_LabAgent"
$Py = ".\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
```

### 1. Start the broken container and Prometheus

```powershell
docker compose up -d
```

`lab-victim` allocates 10 MB every second against a 128 MiB limit. The kernel
kills it roughly every 26 seconds, and Docker restarts it. Forever.

### 2. Start the exporter — leave this window open

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$((Resolve-Path $Py).Path)' -m agent.exporter"
```

A new window opens and prints a line every 5 seconds. It publishes the
container's numbers on port 9101; Prometheus scrapes them.

### 3. Show your partners the problem is real

```powershell
docker inspect lab-victim --format "status={{.State.Status}} restarts={{.RestartCount}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}"
```

Say this out loud: **`restarts` is in the hundreds, yet `exit=0` and
`oom=false`** — Docker wipes those two fields every time it restarts the
container. Reading only this would tell you the container is healthy.

### 4. Wait until Prometheus has fresh data

```powershell
& $Py -m agent.wait_for_metrics
```

Prints `fresh samples ready`. It refuses anything older than 30 seconds,
because Prometheus will happily serve five-minute-old numbers as if they were
current.

### 5. Run the agent — this is the demo

```powershell
& $Py -m agent.run
```

Read the trace aloud, line by line:

| Line | What to say |
|---|---|
| `DETECT` | Plain Python decided there is an incident. No model involved yet. |
| `RECALL` | How many times this happened before, and how many runbook sections were retrieved. |
| `REASON` | The model asks for a tool, and says why. This is the first paid call. |
| `ACT` | The dispatcher ran it — only because the allowlist permitted it. |
| `REASON` | It reads the result and decides again. |
| `RECORD` | The incident goes into memory, so the next run knows. |
| `VERDICT` | The diagnosis, with a confidence. |
| last line | Tool calls, model calls, **tokens**, milliseconds. That is the cost. |

### 6. Show what it remembered

```powershell
& $Py -c "from agent.adapters.memory_sqlite import SqliteMemory; rows = SqliteMemory().recent_incidents('RB-002', hours=24); print('prior incidents:', len(rows)); [print(r['ts'], '|', r['diagnosis']) for r in rows]"
```

Run step 5 again afterwards and `RECALL` counts one higher. That is the
difference between a script and an agent.

---

## The two moments worth pausing on

**It refuses things.** If the model asks to restart a container that is not
`lab-*`, Python refuses and tells it why — the run continues:

```powershell
& $Py -c "from tests.test_graph_mock import FORBIDDEN_REQUEST, run; from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM; final, llm, mem = run(llm=MockLLM([FORBIDDEN_REQUEST, DEFAULT_SCRIPT[1]])); print([l for l in final['history'] if 'REFUSED' in l][0])"
```

Free — no model call. The allowlist, not the prompt, is what stops it.

**The corpus decides the quality.** Same retriever, same question, three
different document sets:

```powershell
& $Py -m agent.corpus_experiment
```

Free. The third corpus answers a container question with `🚀 Challenge` — no
error, no warning. That is the failure nobody notices.

---

## If it fails

| Message | Fix |
|---|---|
| `no lab_container_* series` | the exporter window is closed — redo step 2 |
| `samples are N s old` | same thing: the exporter died, Prometheus is serving stale data |
| `LLMUnavailable ... 401/403` | credentials in `.env` |
| `docker: error during connect` | Docker Desktop is not running |

## When you are done

```powershell
docker compose down
```

Close the exporter window with Ctrl+C. `lab_agent.db` stays — it is the agent's
memory. Delete it to reset the demo to "0 prior incidents".
