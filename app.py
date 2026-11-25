import asyncio
import errno
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
COPILOT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
MAX_COPILOT_RETRIES = 3
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
TOOL_RETRY_INSTRUCTIONS = """

For failed tool actions, make no more than 3 retries total. After the third retry,
stop and report the blocker instead of trying more commands or workarounds.
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


def command_name(path: Path) -> str:
    content = path.read_text()
    frontmatter = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", content, re.DOTALL)
    if frontmatter:
        name = re.search(r"^name:\s*[\"']?([^\"'\n]+)", frontmatter.group(1), re.MULTILINE)
        if name:
            return name.group(1).strip()
    return path.parent.name if path.name == "SKILL.md" else path.stem


def discover_copilot_commands(workdir: str = "") -> dict[str, list[str]]:
    home = Path.home()
    skill_roots = [home / ".copilot" / "skills", home / ".agents" / "skills"]
    agent_roots = [home / ".copilot" / "agents"]
    if workdir:
        root = Path(workdir).expanduser()
        skill_roots.extend(root / path for path in (".github/skills", ".agents/skills", ".claude/skills"))
        agent_roots.append(root / ".github" / "agents")

    skills = {
        command_name(path)
        for root in skill_roots
        if root.is_dir()
        for path in root.glob("*/SKILL.md")
    }
    agents = {
        command_name(path)
        for root in agent_roots
        if root.is_dir()
        for path in root.glob("*.md")
    }
    return {
        "skills": sorted(skills, key=str.casefold),
        "agents": sorted(agents, key=str.casefold),
    }


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


def parse_copilot_prompt(prompt: str) -> tuple[list[str], str]:
    parts = prompt.strip().split(maxsplit=2)
    if not parts or not parts[0].startswith("/"):
        return [], prompt

    command = parts[0][1:]
    if command == "agent":
        if len(parts) < 2:
            return [], prompt
        agent = parts[1]
        task = parts[2] if len(parts) > 2 else ""
        return ["--agent", agent], task

    if command == "skill":
        if len(parts) < 2:
            return [], prompt
        skill = parts[1]
        task = parts[2] if len(parts) > 2 else ""
    else:
        skill = command
        task = parts[1] + (f" {parts[2]}" if len(parts) > 2 else "")

    skill_prompt = f"Use the /{skill} skill"
    if task:
        skill_prompt += f" to complete this task:\n{task}"
    return [], skill_prompt


async def run_job(run: dict, job: dict, context: str, outputs: dict[str, str]) -> None:
    job["status"] = "running"
    job["started_at"] = time.time()

    if job["type"] == "copilot":
        command_args, prompt = parse_copilot_prompt(job["prompt"])
        prompt += TOOL_RETRY_INSTRUCTIONS + STATUS_INSTRUCTIONS
        if context:
            prompt += "\n\nResults from earlier stages:\n\n" + context
        argv = [
            COPILOT_CMD,
            *command_args,
            "--model",
            COPILOT_MODEL,
            "-p",
            prompt,
            "--allow-all-tools",
        ]
    else:
        argv = [sys.executable, "-u", "-c", job["script"]]

    env = {
        **os.environ,
        "FACTORY_RUN_ID": run["id"],
        "FACTORY_WORKDIR": run["workdir"],
        **outputs,
    }
    max_attempts = MAX_COPILOT_RETRIES + 1 if job["type"] == "copilot" else 1
    for attempt in range(max_attempts):
        if run["status"] == "canceled":
            job["status"] = "canceled"
            job["finished_at"] = time.time()
            return
        if attempt:
            job["log"] += f"\n--- Copilot retry {attempt}/{MAX_COPILOT_RETRIES} ---\n"
        attempt_log = ""
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
            attempt_log = f"failed to start {argv[0]}: {exc}\n"
            job["log"] += attempt_log
            job["verdict"] = FAILED
            job["finished_at"] = time.time()
            job["status"] = "failed"
            return

        procs = RUN_PROCS.setdefault(run["id"], set())
        procs.add(proc)
        async for line in proc.stdout:
            decoded = line.decode(errors="replace")
            attempt_log += decoded
            job["log"] += decoded
        rc = await proc.wait()
        procs.discard(proc)
        job["exit_code"] = rc

        if run["status"] == "canceled":
            job["status"] = "canceled"
            job["finished_at"] = time.time()
            return
        attempt_result = {**job, "exit_code": rc, "log": attempt_log}
        job["verdict"] = job_verdict(attempt_result)
        if job["type"] != "copilot" or job["verdict"] != FAILED:
            break
    job["finished_at"] = time.time()
    job["status"] = "success" if job["verdict"] == APPROVED else "failed"


async def execute_run(run: dict, start: int = 0) -> None:
    run["status"] = "running"
    run["started_at"] = time.time()
    workdir = Path(run["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    failed = False

    for i, stage in enumerate(run["stages"]):
        if i < start:
            continue
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


def start_run_task(run: dict, start: int = 0) -> None:
    task = asyncio.create_task(execute_run(run, start))
    RUN_TASKS.add(task)
    task.add_done_callback(RUN_TASKS.discard)


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
    start_run_task(run)
    return run


app = FastAPI(title="Factory")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/pipelines")
def list_pipelines() -> list[Pipeline]:
    return list(PIPELINES.values())


@app.get("/api/copilot-commands")
def list_copilot_commands(workdir: str = "") -> dict[str, list[str]]:
    return discover_copilot_commands(workdir.strip())


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


@app.delete("/api/runs")
def clear_runs() -> dict[str, int]:
    runs_dir = RUNS_DIR.resolve()
    cleared = 0
    for run_id, run in list(RUNS.items()):
        if run["status"] not in TERMINAL:
            continue
        run_dir = (RUNS_DIR / run_id).resolve()
        if runs_dir not in run_dir.parents:
            raise HTTPException(500, f"invalid run directory for '{run_id}'")
        try:
            (run_dir / "run.json").unlink(missing_ok=True)
            run_dir.rmdir()
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT):
                raise
        del RUNS[run_id]
        cleared += 1
    return {"deleted": cleared}


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


@app.post("/api/runs/{rid}/stages/{si}/retry")
async def retry_stage(rid: str, si: int) -> dict:
    run = RUNS.get(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    if not 0 <= si < len(run["stages"]):
        raise HTTPException(404, "stage not found")
    if run["status"] not in TERMINAL:
        raise HTTPException(409, f"run is {run['status']}, can only retry a finished run")

    pipeline = PIPELINES.get(run["pipeline_id"])
    if pipeline is None:
        raise HTTPException(409, "the pipeline was deleted")
    stage_names = [stage.name for stage in pipeline.stages]
    stage_name = run["stages"][si]["name"]
    if stage_name not in stage_names:
        raise HTTPException(409, f"stage '{stage_name}' is no longer in the pipeline")

    pipeline_index = stage_names.index(stage_name)
    run["stages"][si:] = [new_stage(stage) for stage in pipeline.stages[pipeline_index:]]
    run.update(status="running", finished_at=None, pipeline_name=pipeline.name)
    save_run(run)
    start_run_task(run, si)
    return run


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
