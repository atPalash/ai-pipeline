# ai-pipeline

A minimal pipeline runner with a web UI. Stages run sequentially; jobs in a stage run in parallel. Jobs are Python snippets or GitHub Copilot CLI prompts.

## Installation

```
pip install -r requirements.txt
```

Requires Python 3 with the packages above and, for `copilot` jobs, the Copilot CLI on `PATH` (or set `COPILOT_CMD`).

## Usage

```
python app.py                # serves http://127.0.0.1:8000 (override with PORT)
```

Create and edit pipelines in the browser, then press **Run**. Pipelines are stored in `pipelines.json`; runs are stored under `runs/`. Each run gets its own folder unless the pipeline sets a working directory.

### Job verdicts

Copilot jobs must end their reply with `STATUS: APPROVED` or `STATUS: REJECTED`. Python jobs pass when they exit 0.

Earlier job output is passed to Python jobs as `FACTORY_OUT_<STAGE>_<JOB>` environment variables and included in later Copilot prompts.

