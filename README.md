# Budget Agent

One personal budget assistant built twice — `API/` uses a hand-written raw Chat Completions harness; `SDK/` uses the OpenAI Agents SDK. Both run the same scenario, and the results are compared. See `REPORT.md` for the full analysis.

## Setup

```
pip install -r requirements.txt
```

Set three environment variables. **Never put the key in code or commit it to git — environment variable only.**

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | API key |
| `OPENAI_BASE_URL` | no | Base URL (default: OpenAI) |
| `MODEL_NAME` | no | Model name (default: `gpt-4o-mini`) |

**OpenAI:**
```
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

**Google Gemini (OpenAI-compatible endpoint):**
```
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-2.5-flash-lite
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY  = "your-key-here"
$env:OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:MODEL_NAME      = "gemini-2.5-flash-lite"
$env:PYTHONUTF8      = "1"
```

`PYTHONUTF8=1` is needed on Windows to avoid cp1255 encoding errors in terminal output.

## Commands

```powershell
# Offline tests — no API key needed
python -m pytest -v

# Run both harnesses against the scenario
python API/main.py --scenario scenario.json --out results/api.json
python SDK/main.py --scenario scenario.json --out results/sdk.json

# Compare results side by side
python scripts/compare.py results/api.json results/sdk.json
```

Each run wipes `shared/data/` before starting so results are always from a clean state. Free-tier rate limits require a pause between calls; a 4-second delay is built in before every model call.

## Project structure

| Path | Purpose |
|---|---|
| `API/` | Raw Chat Completions harness (loop, context, history, compaction, caching, tokens) |
| `SDK/` | OpenAI Agents SDK version (Runner, SQLiteSession, function tools) |
| `shared/` | Domain tools, file-based data store, memory, pricing, schemas, scenario runner |
| `tests/` | Offline pytest suite (67 tests, no API key needed) |
| `scenario.json` | 7-turn, 2-session benchmark scenario with 7 correctness checks |
| `results/` | JSON output from each harness run |
| `scripts/compare.py` | Side-by-side token / cost / check comparison of two result files |
| `REPORT.md` | Full analysis: token counts, cost, harness complexity, tradeoffs |
