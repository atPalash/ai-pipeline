# ai-pipeline

A minimal GitLab-CI-like pipeline runner with a web UI. Stages run in order, jobs within a stage run in parallel, and each job is either a Python snippet or a GitHub Copilot CLI prompt.

## Installation

```
pip install -r requirements.txt
```

Requires Python 3 with the packages above and, for `copilot` jobs, the Copilot CLI on `PATH` (or set `COPILOT_CMD`).

## Usage

```
python app.py                # serves http://127.0.0.1:8000 (override with PORT)
```

Create and edit pipelines in the browser, then press **Run**. Pipelines are stored in `pipelines.json`; each run gets its own folder under `runs/` unless the pipeline sets a working directory.

### Job verdicts

Copilot jobs must end their reply with one of:

- `STATUS: APPROVED` - the stage continues.
- `STATUS: WAITING_FOR_MANUAL_APPROVAL` - the run waits for you to Approve or Reject in the UI.
- `STATUS: REJECTED` - the pipeline stops.

Python jobs count as approved when they exit 0 without a STATUS line. Earlier job output is available to later jobs as `FACTORY_OUT_<JOB_NAME>` environment variables (Python) or in the prompt context (Copilot).

