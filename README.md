# ai-pipeline

A minimal pipeline runner with a web UI. Stages run sequentially; jobs in a stage run in parallel. Jobs are terminal commands or GitHub Copilot CLI prompts.

## Installation

```
pip install -r requirements.txt
```

Requires Python 3 with the packages above and, for `copilot` jobs, the Copilot CLI on `PATH` (or set `COPILOT_CMD`).
Copilot jobs select the OpenRouter model `deepseek/deepseek-v4.1-flash-1017`; configure the CLI's OpenRouter provider credentials before running them. OpenRouter charges for usage.
Copilot calls that exit unsuccessfully or omit a required status are retried up to 3 times (4 calls total); an explicit `STATUS: REJECTED` is not retried. Copilot is also instructed to stop after 3 failed tool retries within a call, but that instruction is advisory.
Copilot can pause for clarification by ending with `STATUS: WAIT - <question>`. Copy the displayed CLI command and answer in that Copilot CLI session; its messages appear in the waiting job log. The run view enables **Approve** after a new user message, and only continues the pipeline when that button is clicked.

Copilot job prompts can invoke skills with `/skill <name> <task>` (or the shorthand `/name <task>`). Invoke a custom agent with `/agent <name> <task>`; the runner selects it through Copilot CLI's `--agent` option. Skills and agents must already be available to Copilot CLI from the job's working directory or user configuration.

## Usage

```
python app.py                # serves http://127.0.0.1:8000 (override with PORT)
```

Create and edit pipelines in the browser, then press **Run**. Pipelines are stored in `pipelines.json`; runs are stored under `runs/`. Each run gets its own folder unless the pipeline sets a working directory.

### Job verdicts

Copilot jobs must end their reply with `STATUS: APPROVED` or `STATUS: REJECTED`. Terminal jobs use the `terminal` field for a Bash command and pass when it exits 0. Commands run in the pipeline working directory and can invoke tools available on the host, e.g. `python script.py` or `g++ main.cpp -o main && ./main`.
Copilot jobs default to a five-minute timeout, configurable per job in minutes. On timeout, the process is stopped and the run pauses for user input; resume the CLI session, send guidance, then approve to continue.

Earlier job output is passed to terminal commands as `FACTORY_OUT_<STAGE>_<JOB>` environment variables and included in later Copilot prompts.

