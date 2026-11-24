import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "pipelines.json"
RUNS_DIR = BASE_DIR / "runs"
COPILOT_CMD = os.environ.get("COPILOT_CMD", "copilot")
TERMINAL = {"success", "failed", "canceled"}

APPROVED = "APPROVED"
REJECTED = "REJECTED"
FAILED = "FAILED"
STATUS_RE = re.compile(rf"STATUS:\W*({APPROVED}|{REJECTED})\b")
STATUS_INSTRUCTIONS = f"""

End your reply with exactly one of these lines:
STATUS: {APPROVED} - the task is done
STATUS: {REJECTED} - the task cannot be done or is unacceptable
Do not invent results or work around denied access.
"""


class Job(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    type: Literal["python", "copilot"] = "python"
    script: str = "print('done')"
    prompt: str = ""

    @model_validator(mode="after")
    def validate_prompt(self):
        if self.type == "copilot" and not self.prompt.strip():
            raise ValueError(f"copilot job '{self.name}' needs a prompt")
        return self


class Stage(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    jobs: list[Job] = Field(min_length=1)


class PipelineIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    workdir: str = ""
    stages: list[Stage] = Field(min_length=1)

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, value: str) -> str:
        value = value.strip()
        if value and not Path(value).is_dir():
            raise ValueError(f"workdir '{value}' is not an existing directory")
        return value


class Pipeline(PipelineIn):
    id: str


def load_pipelines() -> dict[str, Pipeline]:
    if not DATA_FILE.exists():
        return {}
    try:
        raw = json.loads(DATA_FILE.read_text())
        return {p["id"]: Pipeline(**p) for p in raw}
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return {}


def save_pipelines() -> None:
    DATA_FILE.write_text(json.dumps([p.model_dump() for p in PIPELINES.values()], indent=2))


def save_run(run: dict) -> None:
    path = RUNS_DIR / run["id"] / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))


def load_runs() -> dict[str, dict]:
    runs = {}
    for path in RUNS_DIR.glob("*/run.json"):
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if run["status"] not in TERMINAL:
            run["status"] = "canceled"
            run["finished_at"] = run.get("finished_at") or time.time()
            for stage in run["stages"]:
                if stage["status"] not in ("success", "failed", "skipped"):
                    stage["status"] = "canceled"
                for job in stage["jobs"]:
                    if job["status"] not in ("success", "failed", "skipped"):
                        job["status"] = "canceled"
        runs[run["id"]] = run
    return runs


PIPELINES = load_pipelines()
RUNS = load_runs()
RUN_PROCS: dict[str, set[asyncio.subprocess.Process]] = {}
RUN_TASKS: set[asyncio.Task] = set()


def new_stage(stage: Stage) -> dict:
    return {
        "name": stage.name,
        "status": "created",
        "verdict": None,
        "jobs": [
            {
                "name": job.name,
                "type": job.type,
                "script": job.script,
                "prompt": job.prompt,
                "status": "created",
                "verdict": None,
                "log": "",
                "exit_code": None,
                "started_at": None,
                "finished_at": None,
            }
            for job in stage.jobs
        ],
    }


def build_context(run: dict, upto: int) -> str:
    parts = []
    for stage in run["stages"][:upto]:
        parts.append(f"## Stage: {stage['name']}")
        for job in stage["jobs"]:
            parts.append(
                f"### Job: {job['name']}\n"
                f"Status: {job['verdict'] or job['status']}\n"
                f"```\n{job['log'].strip()[-4000:]}\n```"
            )
    return "\n\n".join(parts)


def output_env(run: dict, upto: int) -> dict[str, str]:
    env = {}
    for stage in run["stages"][:upto]:
        for job in stage["jobs"]:
            key = re.sub(r"\W", "_", f"{stage['name']}_{job['name']}").upper()
            env[f"FACTORY_OUT_{key}"] = job["log"].strip()[-4000:]
    return env


def signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def job_verdict(job: dict) -> str:
    if job["exit_code"] != 0:
        return FAILED
    if job["type"] == "copilot":
        found = STATUS_RE.findall(job["log"])
        if found:
            return found[-1]
        return FAILED
    return APPROVED


async def run_job(run: dict, job: dict, context: str, outputs: dict[str, str]) -> None:
    job["status"] = "running"
    job["started_at"] = time.time()

    if job["type"] == "copilot":
        prompt = job["prompt"] + STATUS_INSTRUCTIONS
        if context:
            prompt += "\n\nResults from earlier stages:\n\n" + context
        argv = [COPILOT_CMD, "-p", prompt, "--allow-all-tools"]
    else:
        argv = [sys.executable, "-u", "-c", job["script"]]

    env = {
        **os.environ,
        "FACTORY_RUN_ID": run["id"],
        "FACTORY_WORKDIR": run["workdir"],
        **outputs,
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=run["workdir"],
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        job["log"] = f"failed to start {argv[0]}: {exc}\n"
        job["status"] = "failed"
        job["verdict"] = FAILED
        job["finished_at"] = time.time()
        return

    procs = RUN_PROCS.setdefault(run["id"], set())
    procs.add(proc)
    async for line in proc.stdout:
        job["log"] += line.decode(errors="replace")
    rc = await proc.wait()
    procs.discard(proc)
    job["exit_code"] = rc
    job["finished_at"] = time.time()

    if run["status"] == "canceled":
        job["status"] = "canceled"
        return
    job["verdict"] = job_verdict(job)
    job["status"] = "success" if job["verdict"] == APPROVED else "failed"


async def execute_run(run: dict) -> None:
    run["status"] = "running"
    run["started_at"] = time.time()
    workdir = Path(run["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    failed = False

    for i, stage in enumerate(run["stages"]):
        if run["status"] == "canceled":
            break
        if failed:
            stage["status"] = "skipped"
            for job in stage["jobs"]:
                job["status"] = "skipped"
            continue

        stage["status"] = "running"
        context = build_context(run, i)
        outputs = output_env(run, i)
        await asyncio.gather(
            *(run_job(run, job, context, outputs) for job in stage["jobs"])
        )

        if run["status"] == "canceled":
            stage["status"] = "canceled"
            break

        verdicts = {job["verdict"] for job in stage["jobs"]}
        if REJECTED in verdicts:
            stage["verdict"] = REJECTED
        elif FAILED in verdicts:
            stage["verdict"] = FAILED
        else:
            stage["verdict"] = APPROVED
        stage["status"] = "success" if stage["verdict"] == APPROVED else "failed"
        failed = stage["verdict"] != APPROVED
        save_run(run)

    if run["status"] == "canceled":
        for stage in run["stages"]:
            if stage["status"] in ("created", "running"):
                stage["status"] = "canceled"
            for job in stage["jobs"]:
                if job["status"] in ("created", "running"):
                    job["status"] = "canceled"
    else:
        run["status"] = "failed" if failed else "success"
    run["finished_at"] = time.time()
    RUN_PROCS.pop(run["id"], None)
    save_run(run)


def create_run(pipeline: Pipeline) -> dict:
    run_id = uuid.uuid4().hex[:8]
    run = {
        "id": run_id,
        "pipeline_id": pipeline.id,
        "pipeline_name": pipeline.name,
        "workdir": pipeline.workdir or str(RUNS_DIR / run_id / "workspace"),
        "status": "pending",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "stages": [new_stage(stage) for stage in pipeline.stages],
    }
    RUNS[run_id] = run
    save_run(run)
    task = asyncio.create_task(execute_run(run))
    RUN_TASKS.add(task)
    task.add_done_callback(RUN_TASKS.discard)
    return run


app = FastAPI(title="Factory")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/pipelines")
def list_pipelines() -> list[Pipeline]:
    return list(PIPELINES.values())


@app.get("/api/pipelines/{pid}")
def get_pipeline(pid: str) -> Pipeline:
    if pid not in PIPELINES:
        raise HTTPException(404, "pipeline not found")
    return PIPELINES[pid]


@app.post("/api/pipelines", status_code=201)
def create_pipeline(body: PipelineIn) -> Pipeline:
    pipeline = Pipeline(id=uuid.uuid4().hex[:8], **body.model_dump())
    PIPELINES[pipeline.id] = pipeline
    save_pipelines()
    return pipeline


@app.put("/api/pipelines/{pid}")
def update_pipeline(pid: str, body: PipelineIn) -> Pipeline:
    if pid not in PIPELINES:
        raise HTTPException(404, "pipeline not found")
    PIPELINES[pid] = Pipeline(id=pid, **body.model_dump())
    save_pipelines()
    return PIPELINES[pid]


@app.delete("/api/pipelines/{pid}", status_code=204)
def delete_pipeline(pid: str) -> None:
    if PIPELINES.pop(pid, None) is None:
        raise HTTPException(404, "pipeline not found")
    save_pipelines()


@app.post("/api/pipelines/{pid}/run", status_code=201)
async def run_pipeline(pid: str) -> dict:
    if pid not in PIPELINES:
        raise HTTPException(404, "pipeline not found")
    return create_run(PIPELINES[pid])


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return sorted(RUNS.values(), key=lambda run: run["created_at"], reverse=True)


@app.get("/api/runs/{rid}")
def get_run(rid: str) -> dict:
    if rid not in RUNS:
        raise HTTPException(404, "run not found")
    return RUNS[rid]


@app.post("/api/runs/{rid}/cancel")
def cancel_run(rid: str) -> dict:
    run = RUNS.get(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    if run["status"] in TERMINAL:
        raise HTTPException(409, "run already finished")
    run["status"] = "canceled"
    for proc in RUN_PROCS.get(rid, ()):
        if proc.returncode is None:
            signal_group(proc, signal.SIGKILL)
    save_run(run)
    return run


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
