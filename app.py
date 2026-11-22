"""Minimal GitLab-CI-like pipeline runner.

Stages run sequentially; jobs within a stage run in parallel.
A job is either a Python snippet or a Copilot CLI prompt, run in a subprocess.
All jobs of a run share one working directory; the orchestrator hands each
stage the outputs and files produced by the earlier stages.
"""

import asyncio
import json
import os
import re
import shlex
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
DEFAULT_SCRIPT = "import time\nprint('working...')\ntime.sleep(5)\nprint('done')"
COPILOT_CMD = os.environ.get("COPILOT_CMD", "copilot")
TERMINAL = {"success", "failed", "canceled"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}
MAX_FILE_CHARS = 20_000
MAX_OUTPUT_CHARS = 4_000
SESSION_RE = re.compile(r"--resume[= ]([\w-]+)")
# Every job ends with a verdict; a stage proceeds only when all its jobs are APPROVED.
APPROVED, WAITING, REJECTED, FAILED = "APPROVED", "WAITING_FOR_MANUAL_APPROVAL", "REJECTED", "FAILED"
STATUS_RE = re.compile(rf"STATUS:\W*({APPROVED}|{WAITING}|{REJECTED})\b")
STATUS_INSTRUCTIONS = f"""

---
You are one stage of an automated pipeline. End your reply with exactly one of these lines:
STATUS: {APPROVED}  (the task is done and the pipeline may continue)
STATUS: {WAITING}  (a human must answer your questions, review, or decide)
STATUS: {REJECTED}  (the task cannot be done or the result is not acceptable)
If you need the user's input, ask your questions and use {WAITING}; do not invent answers.
The user may continue this session in a terminal; once they are satisfied, write the agreed
result to `{{outcome_file}}` in the current directory."""


class Job(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    type: Literal["python", "copilot"] = "python"
    script: str = DEFAULT_SCRIPT
    prompt: str = ""

    @model_validator(mode="after")
    def _check(self):
        if self.type == "copilot" and not self.prompt.strip():
            raise ValueError(f"copilot job '{self.name}' needs a prompt")
        return self


class Stage(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    jobs: list[Job] = Field(min_length=1)


class PipelineIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    # Empty -> a fresh directory per run under runs/<run_id>/workspace.
    workdir: str = ""
    stages: list[Stage] = Field(min_length=1)

    @field_validator("workdir")
    @classmethod
    def _check_workdir(cls, v: str) -> str:
        v = v.strip()
        if v and not Path(v).is_dir():
            raise ValueError(f"workdir '{v}' is not an existing directory")
        return v


class Pipeline(PipelineIn):
    id: str


class Decision(BaseModel):
    action: Literal["approve", "reject"]


# ---------------------------------------------------------------- storage

def _sample_pipeline() -> Pipeline:
    sleep = lambda s, msg: f"import time\nprint('{msg}')\ntime.sleep({s})\nprint('ok')"
    return Pipeline(
        id=uuid.uuid4().hex[:8],
        name="sample",
        stages=[
            Stage(name="build", jobs=[Job(name="compile", script=sleep(5, "compiling"))]),
            Stage(name="test", jobs=[
                Job(name="unit", script=sleep(4, "unit tests")),
                Job(name="integration", script=sleep(7, "integration tests")),
                Job(name="lint", script=sleep(2, "linting")),
            ]),
            Stage(name="deploy", jobs=[Job(name="deploy", script=sleep(5, "deploying"))]),
        ],
    )


def load_pipelines() -> dict[str, Pipeline]:
    if not DATA_FILE.exists():
        p = _sample_pipeline()
        return {p.id: p}
    raw = json.loads(DATA_FILE.read_text())
    return {p["id"]: Pipeline(**p) for p in raw}


def save_pipelines() -> None:
    DATA_FILE.write_text(json.dumps([p.model_dump() for p in PIPELINES.values()], indent=2))


PIPELINES: dict[str, Pipeline] = load_pipelines()
RUNS: dict[str, dict] = {}
RUN_PROCS: dict[str, set[asyncio.subprocess.Process]] = {}
RUN_TASKS: set[asyncio.Task] = set()
WAITERS: dict[int, asyncio.Event] = {}  # id(stage dict) -> set when the user decides


# ---------------------------------------------------------------- context

def snapshot(root: Path) -> dict[str, float]:
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath, f)
            try:
                files[str(p.relative_to(root))] = p.stat().st_mtime
            except OSError:
                pass
    return files


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "...(truncated)\n" + text[-limit:]


def build_context(run: dict, upto: int) -> str:
    """Markdown summary of stages [0, upto): job outputs and the files each stage wrote."""
    workdir = Path(run["workdir"])
    parts = []
    for stage in run["stages"][:upto]:
        parts.append(f"## Stage: {stage['name']} ({stage['verdict'] or stage['status']})")
        for job in stage["jobs"]:
            parts.append(f"### Job: {job['name']} ({job['verdict'] or job['status']})\nOutput:\n```\n"
                         f"{_tail(job['log'].strip(), MAX_OUTPUT_CHARS)}\n```")
        for rel in stage["files"]:
            try:
                content = (workdir / rel).read_text()
            except (OSError, UnicodeDecodeError):
                parts.append(f"### File: {rel} (binary or unreadable, not shown)")
                continue
            if len(content) > MAX_FILE_CHARS:
                parts.append(f"### File: {rel} ({len(content)} chars, too large, not shown)")
            else:
                parts.append(f"### File: {rel}\n```\n{content}\n```")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- runner

def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the job's whole process group (the job runs in its own session)."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def output_env(run: dict, upto: int) -> dict[str, str]:
    """FACTORY_OUT_<JOB_NAME> = output of each job in stages [0, upto)."""
    return {
        "FACTORY_OUT_" + re.sub(r"\W", "_", job["name"]).upper(): _tail(job["log"].strip(), MAX_OUTPUT_CHARS)
        for stage in run["stages"][:upto]
        for job in stage["jobs"]
    }


def job_verdict(job: dict) -> str:
    if job["exit_code"] != 0:
        return FAILED
    if found := STATUS_RE.findall(job["log"]):
        return found[-1]
    if job["type"] == "copilot":
        job["log"] += f"\n[factory] no STATUS line in the reply; defaulting to {WAITING}\n"
        return WAITING
    return APPROVED


async def wait_for_approval(run: dict, stage: dict) -> str:
    stage["status"] = "waiting"
    run["status"] = "waiting"
    event = asyncio.Event()
    WAITERS[id(stage)] = event
    try:
        await event.wait()
    finally:
        WAITERS.pop(id(stage), None)
    if run["status"] == "waiting":
        run["status"] = "running"
    return {"approve": APPROVED, "reject": REJECTED}.get(stage["decision"])


async def run_job(run: dict, job: dict, context: str, outputs: dict[str, str]) -> None:
    job["status"] = "running"
    job["started_at"] = time.time()
    if job["type"] == "copilot":
        prompt = job["prompt"] + STATUS_INSTRUCTIONS.format(outcome_file=job["outcome_file"])
        if context:
            prompt += ("\n\n---\nResults from earlier pipeline stages "
                       "(files are in the current directory):\n\n" + context)
        job["prompt_sent"] = prompt
        argv = [COPILOT_CMD, "-p", prompt, "--allow-all-tools"]
    else:
        argv = [sys.executable, "-u", "-c", job["script"]]
    env = {
        **os.environ,
        "FACTORY_RUN_ID": run["id"],
        "FACTORY_WORKDIR": run["workdir"],
        "FACTORY_CONTEXT": str(RUNS_DIR / run["id"] / "context.md"),
        **outputs,
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=run["workdir"], env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as e:
        job["log"] += f"failed to start {argv[0]}: {e}\n"
        job["status"] = "failed"
        job["verdict"] = FAILED
        job["finished_at"] = time.time()
        return
    procs = RUN_PROCS.setdefault(run["id"], set())
    procs.add(proc)
    if run["status"] == "paused":
        # Paused while the process was being spawned.
        _signal_group(proc, signal.SIGSTOP)
        job["status"] = "paused"
    async for line in proc.stdout:
        job["log"] += line.decode(errors="replace")
    rc = await proc.wait()
    procs.discard(proc)
    job["exit_code"] = rc
    if job["type"] == "copilot" and (ids := SESSION_RE.findall(job["log"])):
        job["session_id"] = ids[-1]
        job["resume_cmd"] = f"cd {shlex.quote(run['workdir'])} && copilot --resume={ids[-1]}"
    job["finished_at"] = time.time()
    if run["status"] == "canceled":
        job["status"] = "canceled"
        return
    job["verdict"] = job_verdict(job)
    job["status"] = {APPROVED: "success", WAITING: "waiting"}.get(job["verdict"], "failed")


async def execute_run(run: dict) -> None:
    if run["status"] == "pending":
        run["status"] = "running"
    run["started_at"] = time.time()
    workdir = Path(run["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    context_file = RUNS_DIR / run["id"] / "context.md"
    context_file.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    for i, stage in enumerate(run["stages"]):
        while run["status"] == "paused":
            await asyncio.sleep(0.5)
        if run["status"] == "canceled":
            break
        if failed:
            stage["status"] = "skipped"
            for job in stage["jobs"]:
                job["status"] = "skipped"
            continue
        stage["status"] = "running"
        context = build_context(run, i)
        context_file.write_text(context)
        outputs = output_env(run, i)
        before = await asyncio.to_thread(snapshot, workdir)
        await asyncio.gather(*(run_job(run, j, context, outputs) for j in stage["jobs"]))
        if run["status"] != "canceled":
            verdicts = {j["verdict"] for j in stage["jobs"]}
            if verdicts & {REJECTED, FAILED}:
                stage["verdict"] = REJECTED if REJECTED in verdicts else FAILED
            elif WAITING in verdicts:
                stage["verdict"] = WAITING
                stage["verdict"] = await wait_for_approval(run, stage)
                for j in stage["jobs"]:
                    if j["status"] == "waiting" and run["status"] != "canceled":
                        j["status"] = "success" if stage["verdict"] == APPROVED else "failed"
            else:
                stage["verdict"] = APPROVED
        # After the approval wait, so files written while chatting in the terminal count too.
        after = await asyncio.to_thread(snapshot, workdir)
        stage["files"] = sorted(p for p, m in after.items() if before.get(p) != m)
        if run["status"] == "canceled":
            stage["status"] = "canceled"
            break
        stage["status"] = "success" if stage["verdict"] == APPROVED else "failed"
        failed = failed or stage["verdict"] != APPROVED

    if run["status"] == "canceled":
        for stage in run["stages"]:
            if stage["status"] in ("created", "pending"):
                stage["status"] = "canceled"
            for job in stage["jobs"]:
                if job["status"] in ("created", "pending", "waiting"):
                    job["status"] = "canceled"
    else:
        run["status"] = "failed" if failed else "success"
    run["finished_at"] = time.time()
    RUN_PROCS.pop(run["id"], None)


def create_run(p: Pipeline) -> dict:
    run_id = uuid.uuid4().hex[:8]
    run = {
        "id": run_id,
        "pipeline_id": p.id,
        "pipeline_name": p.name,
        "workdir": p.workdir or str(RUNS_DIR / run_id / "workspace"),
        "status": "pending",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "stages": [
            {
                "name": s.name,
                "status": "created",
                "verdict": None,
                "decision": None,
                "files": [],
                "jobs": [
                    {"name": j.name, "type": j.type, "script": j.script, "prompt": j.prompt,
                     "outcome_file": re.sub(r"\W", "_", j.name) + ".md",
                     "session_id": None, "resume_cmd": None, "verdict": None,
                     "prompt_sent": None, "status": "created", "log": "",
                     "exit_code": None, "started_at": None, "finished_at": None}
                    for j in s.jobs
                ],
            }
            for s in p.stages
        ],
    }
    RUNS[run["id"]] = run
    task = asyncio.create_task(execute_run(run))
    RUN_TASKS.add(task)
    task.add_done_callback(RUN_TASKS.discard)
    return run


# ---------------------------------------------------------------- API

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
    p = Pipeline(id=uuid.uuid4().hex[:8], **body.model_dump())
    PIPELINES[p.id] = p
    save_pipelines()
    return p


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
    return sorted(RUNS.values(), key=lambda r: r["created_at"], reverse=True)


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
            _signal_group(proc, signal.SIGKILL)
    for stage in run["stages"]:
        if (event := WAITERS.get(id(stage))) is not None:
            stage["decision"] = "cancel"
            event.set()
    return run


@app.post("/api/runs/{rid}/stages/{si}/decision")
def decide_stage(rid: str, si: int, body: Decision) -> dict:
    run = RUNS.get(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    if not 0 <= si < len(run["stages"]):
        raise HTTPException(404, "stage not found")
    stage = run["stages"][si]
    event = WAITERS.get(id(stage))
    if event is None:
        raise HTTPException(409, f"stage is {stage['status']}, not waiting for approval")
    stage["decision"] = body.action
    event.set()
    return run


@app.post("/api/runs/{rid}/pause")
def pause_run(rid: str) -> dict:
    run = RUNS.get(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    if run["status"] not in ("pending", "running"):
        raise HTTPException(409, f"run is {run['status']}, cannot pause")
    run["status"] = "paused"
    for stage in run["stages"]:
        for job in stage["jobs"]:
            if job["status"] == "running":
                job["status"] = "paused"
    for proc in RUN_PROCS.get(rid, ()):
        if proc.returncode is None:
            _signal_group(proc, signal.SIGSTOP)
    return run


@app.post("/api/runs/{rid}/resume")
def resume_run(rid: str) -> dict:
    run = RUNS.get(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    if run["status"] != "paused":
        raise HTTPException(409, f"run is {run['status']}, cannot resume")
    run["status"] = "running"
    for stage in run["stages"]:
        for job in stage["jobs"]:
            if job["status"] == "paused":
                job["status"] = "running"
    for proc in RUN_PROCS.get(rid, ()):
        if proc.returncode is None:
            _signal_group(proc, signal.SIGCONT)
    return run


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
