# ai-pipeline

A minimal pipeline runner with a web UI. Stages run sequentially; jobs in a stage run in parallel. Jobs are Python snippets or GitHub Copilot CLI prompts.

## Installation

```
pip install -r requirements.txt
```

Requires Python 3 with the packages above and, for `copilot` jobs, the Copilot CLI on `PATH` (or set `COPILOT_CMD`).
Copilot jobs select the OpenRouter model `deepseek/deepseek-v4.1-flash-1017`; configure the CLI's OpenRouter provider credentials before running them. OpenRouter charges for usage.
Copilot calls that exit unsuccessfully or omit a required status are retried up to 3 times (4 calls total); an explicit `STATUS: REJECTED` is not retried. Copilot is also instructed to stop after 3 failed tool retries within a call, but that instruction is advisory.

## Usage

```
python app.py                # serves http://127.0.0.1:8000 (override with PORT)
```

Create and edit pipelines in the browser, then press **Run**. Pipelines are stored in `pipelines.json`; runs are stored under `runs/`. Each run gets its own folder unless the pipeline sets a working directory.

### Job verdicts

Copilot jobs must end their reply with `STATUS: APPROVED` or `STATUS: REJECTED`. Python jobs pass when they exit 0.

Earlier job output is passed to Python jobs as `FACTORY_OUT_<STAGE>_<JOB>` environment variables and included in later Copilot prompts.

