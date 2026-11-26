import asyncio
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import app


class FakeStdout:
    def __init__(self, output: str):
        self.lines = iter(output.splitlines(keepends=True))

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.lines).encode()
        except StopIteration:
            raise StopAsyncIteration


class FakeProcess:
    def __init__(self, output: str, returncode: int):
        self.stdout = FakeStdout(output)
        self.returncode = returncode

    async def wait(self):
        return self.returncode


class HangingProcess:
    def __init__(self):
        self.stdout = FakeStdout("")
        self.returncode = None
        self.finished = asyncio.Event()

    async def wait(self):
        await self.finished.wait()
        return self.returncode


def make_job():
    return {
        "name": "ai",
        "type": "copilot",
        "prompt": "do task",
        "status": "created",
        "log": "",
        "exit_code": None,
        "verdict": None,
        "started_at": None,
        "finished_at": None,
    }


class CopilotRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_quoted_status_example_is_not_a_verdict(self):
        job = make_job()
        job["exit_code"] = 0
        job["log"] = "How do we prove it before emitting `STATUS: APPROVED`?"

        self.assertEqual(app.job_verdict(job), app.FAILED)

    async def test_stops_after_three_retries(self):
        job = make_job()
        run = {
            "id": "test-retry-limit",
            "workdir": tempfile.gettempdir(),
            "status": "running",
        }
        processes = [FakeProcess("temporary failure\n", 1) for _ in range(4)]
        create_process = AsyncMock(side_effect=processes)
        self.addCleanup(app.RUN_PROCS.pop, run["id"], None)

        with patch("app.asyncio.create_subprocess_exec", create_process):
            await app.run_job(run, job, "", {})

        self.assertEqual(create_process.await_count, 4)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["log"].count("Copilot retry"), 3)

    async def test_explicit_rejection_is_not_retried(self):
        job = make_job()
        run = {
            "id": "test-rejection",
            "workdir": tempfile.gettempdir(),
            "status": "running",
        }
        create_process = AsyncMock(
            return_value=FakeProcess("STATUS: REJECTED - blocked\n", 0)
        )
        self.addCleanup(app.RUN_PROCS.pop, run["id"], None)

        with patch("app.asyncio.create_subprocess_exec", create_process):
            await app.run_job(run, job, "", {})

        self.assertEqual(create_process.await_count, 1)
        self.assertEqual(job["verdict"], app.REJECTED)
        self.assertEqual(job["status"], "failed")

    async def test_wait_status_pauses_job_without_retry(self):
        job = make_job()
        run = {
            "id": "test-wait-status",
            "workdir": tempfile.gettempdir(),
            "status": "running",
        }
        create_process = AsyncMock(
            return_value=FakeProcess("What timezone?\nSTATUS: WAIT - What timezone?\n", 0)
        )
        self.addCleanup(app.RUN_PROCS.pop, run["id"], None)

        with patch("app.asyncio.create_subprocess_exec", create_process):
            await app.run_job(run, job, "", {})

        self.assertEqual(create_process.await_count, 1)
        self.assertEqual(job["verdict"], app.WAIT)
        self.assertEqual(job["status"], app.WAITING)
        argv = create_process.await_args.args
        self.assertIn(f"--session-id={job['copilot_session_id']}", argv)


class TerminalCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_terminal_command_in_pipeline_workdir(self):
        with tempfile.TemporaryDirectory() as workdir:
            job = {
                "name": "shell",
                "type": "terminal",
                "terminal": "printf 'hello from terminal\\n'",
                "status": "created",
                "log": "",
                "exit_code": None,
                "verdict": None,
                "started_at": None,
                "finished_at": None,
            }
            run = {"id": "test-command", "workdir": workdir, "status": "running"}

            await app.run_job(run, job, "", {})

        self.assertEqual(job["status"], "success")
        self.assertEqual(job["exit_code"], 0)
        self.assertIn("hello from terminal", job["log"])


class CopilotPromptTests(unittest.TestCase):
    def test_slash_prefix_invokes_skill(self):
        args, prompt = app.parse_copilot_prompt("/aidlc-analysis grill me")

        self.assertEqual(args, [])
        self.assertEqual(prompt, "Use the /aidlc-analysis skill to complete this task:\ngrill me")

    def test_explicit_skill_command_invokes_skill(self):
        args, prompt = app.parse_copilot_prompt("/skill code-review review this diff")

        self.assertEqual(args, [])
        self.assertEqual(prompt, "Use the /code-review skill to complete this task:\nreview this diff")

    def test_agent_command_selects_cli_agent(self):
        args, prompt = app.parse_copilot_prompt("/agent reviewer inspect the changes")

        self.assertEqual(args, ["--agent", "reviewer"])
        self.assertEqual(prompt, "inspect the changes")

    def test_regular_prompt_is_unchanged(self):
        original = "Implement the next feature"

        args, prompt = app.parse_copilot_prompt(original)

        self.assertEqual(args, [])
        self.assertEqual(prompt, original)


class JobValidationTests(unittest.TestCase):
    def test_copilot_timeout_defaults_to_five_minutes(self):
        job = app.Job(name="ai", type="copilot", prompt="do task")

        self.assertEqual(job.timeout_minutes, 5)

    def test_terminal_job_requires_command_text(self):
        with self.assertRaises(ValueError):
            app.Job(name="shell", type="terminal")

    def test_terminal_job_rejects_old_command_field(self):
        with self.assertRaises(ValueError):
            app.Job(name="shell", type="terminal", command="echo done")

    def test_command_is_not_a_job_type(self):
        with self.assertRaises(ValueError):
            app.Job(name="job", type="command")


class CopilotCommandDiscoveryTests(unittest.TestCase):
    def test_discovers_project_and_personal_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workdir = root / "project"
            home = root / "home"
            skill_file = workdir / ".github" / "skills" / "directory-name" / "SKILL.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text("---\nname: project-skill\n---\n")
            agent_file = workdir / ".github" / "agents" / "reviewer.agent.md"
            agent_file.parent.mkdir(parents=True)
            agent_file.write_text("---\nname: reviewer\n---\n")
            personal_skill = home / ".copilot" / "skills" / "personal-skill" / "SKILL.md"
            personal_skill.parent.mkdir(parents=True)
            personal_skill.write_text("---\nname: personal-skill\n---\n")
            home.mkdir(exist_ok=True)

            with patch("app.Path.home", return_value=home):
                commands = app.discover_copilot_commands(str(workdir))

        self.assertIn("project-skill", commands["skills"])
        self.assertIn("personal-skill", commands["skills"])
        self.assertIn("reviewer", commands["agents"])


class WaitingRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_copilot_timeout_stops_job_and_waits_for_user(self):
        with tempfile.TemporaryDirectory() as workdir:
            job = make_job()
            job["timeout_minutes"] = 0.0001
            run = {
                "id": "test-copilot-timeout",
                "workdir": workdir,
                "status": "pending",
                "stages": [
                    {
                        "name": "review",
                        "status": "created",
                        "verdict": None,
                        "jobs": [job],
                    },
                    {
                        "name": "deploy",
                        "status": "created",
                        "verdict": None,
                        "jobs": [make_job()],
                    },
                ],
            }
            proc = HangingProcess()
            create_process = AsyncMock(return_value=proc)

            def stop_process(_proc, _sig):
                proc.returncode = -9
                proc.finished.set()

            self.addCleanup(app.RUN_PROCS.pop, run["id"], None)
            with (
                patch("app.asyncio.create_subprocess_exec", create_process),
                patch("app.signal_group", side_effect=stop_process),
                patch("app.read_copilot_session_events", return_value=[]),
                patch("app.save_run"),
            ):
                await app.execute_run(run)

        self.assertEqual(create_process.await_count, 1)
        self.assertEqual(job["verdict"], app.WAIT)
        self.assertEqual(job["status"], app.WAITING)
        self.assertEqual(job["waiting_reason"], "timeout")
        self.assertEqual(run["status"], app.WAITING)
        self.assertEqual(run["stages"][0]["status"], app.WAITING)
        self.assertEqual(run["stages"][1]["status"], "created")

    async def test_timeout_wait_accepts_cli_guidance_before_approval(self):
        session_id = "922faa19-34da-4008-b993-3d4384670d14"
        with tempfile.TemporaryDirectory() as temp_dir:
            events_path = (
                Path(temp_dir) / "session-state" / session_id / "events.jsonl"
            )
            events_path.parent.mkdir(parents=True)

            def event(event_id, event_type, content):
                return json.dumps(
                    {
                        "id": event_id,
                        "type": event_type,
                        "data": {"content": content},
                    }
                )

            events_path.write_text(
                event("baseline", "assistant.message", "Working") + "\n"
            )
            job = make_job()
            job.update(
                status=app.WAITING,
                verdict=app.WAIT,
                waiting_reason="timeout",
                copilot_session_id=session_id,
                copilot_event_id="baseline",
            )
            stage = {
                "name": "review",
                "status": app.WAITING,
                "verdict": app.WAIT,
                "jobs": [job],
            }
            run = {
                "id": "test-timeout-guidance",
                "status": app.WAITING,
                "finished_at": None,
                "stages": [stage],
            }
            with events_path.open("a") as events:
                events.write(
                    event("guidance", "user.message", "Use a safer approach") + "\n"
                )
                events.write(
                    event("reply", "assistant.message", "I will adjust") + "\n"
                )

            with (
                patch.dict(app.RUNS, {run["id"]: run}),
                patch.object(app, "COPILOT_HOME", Path(temp_dir), create=True),
                patch("app.save_run"),
                patch("app.start_run_task") as start_run,
            ):
                await app.get_run(run["id"])
                self.assertTrue(job["copilot_user_confirmed"])
                self.assertIn("Use a safer approach", job["log"])
                await app.approve_waiting_job(run["id"], 0, 0)

        self.assertEqual(run["status"], "running")
        self.assertEqual(job["status"], "success")
        start_run.assert_called_once_with(run, 0)

    async def test_cli_update_enables_approve_and_is_logged(self):
        session_id = "922faa19-34da-4008-b993-3d4384670d14"
        with tempfile.TemporaryDirectory() as temp_dir:
            events_path = (
                Path(temp_dir) / "session-state" / session_id / "events.jsonl"
            )
            events_path.parent.mkdir(parents=True)

            def event(event_id, event_type, content):
                return json.dumps(
                    {
                        "id": event_id,
                        "type": event_type,
                        "data": {"content": content},
                    }
                )

            wait_event = event(
                "wait", "assistant.message", "Question?\nSTATUS: WAIT - Question?"
            )
            events_path.write_text(wait_event + "\n")
            job = make_job()
            job.update(
                status=app.WAITING,
                verdict=app.WAIT,
                copilot_session_id=session_id,
                copilot_status_event_id="wait",
            )
            stage = {
                "name": "review",
                "status": app.WAITING,
                "verdict": app.WAIT,
                "jobs": [job],
            }
            run = {
                "id": "test-external-session-update",
                "status": app.WAITING,
                "finished_at": None,
                "stages": [stage],
            }
            user_event = event("answer", "user.message", "Pacific time")
            assistant_event = event(
                "reply", "assistant.message", "Thanks, I will use Pacific time."
            )

            with (
                patch.dict(app.RUNS, {run["id"]: run}),
                patch.object(app, "COPILOT_HOME", Path(temp_dir), create=True),
                patch("app.save_run"),
                patch("app.start_run_task") as start_run,
            ):
                with events_path.open("a") as events:
                    events.write(user_event + "\n")
                    events.write(assistant_event + "\n")

                await app.get_run(run["id"])
                self.assertEqual(run["status"], app.WAITING)
                self.assertTrue(job["copilot_user_confirmed"])
                self.assertIn("Pacific time", job["log"])
                self.assertIn("Thanks, I will use Pacific time.", job["log"])
                start_run.assert_not_called()

                await app.approve_waiting_job(run["id"], 0, 0)

        self.assertEqual(run["status"], "running")
        self.assertEqual(job["status"], "success")
        self.assertEqual(job["verdict"], app.APPROVED)
        start_run.assert_called_once_with(run, 0)

    async def test_wait_state_survives_reload(self):
        run = {
            "id": "test-persist-wait",
            "status": app.WAITING,
            "finished_at": None,
            "stages": [
                {
                    "status": app.WAITING,
                    "jobs": [{"status": app.WAITING}],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.RUNS_DIR", Path(temp_dir)):
                app.save_run(run)
                loaded = app.load_runs()[run["id"]]

        self.assertEqual(loaded["status"], app.WAITING)
        self.assertEqual(loaded["stages"][0]["status"], app.WAITING)
        self.assertEqual(loaded["stages"][0]["jobs"][0]["status"], app.WAITING)

    async def test_wait_pauses_later_stages_without_rerunning_completed_jobs(self):
        with tempfile.TemporaryDirectory() as workdir:
            completed = make_job()
            completed.update(status="success", verdict=app.APPROVED, log="finished")
            waiting = make_job()
            run = {
                "id": "test-wait-run",
                "workdir": workdir,
                "status": "pending",
                "stages": [
                    {
                        "name": "review",
                        "status": "created",
                        "verdict": None,
                        "jobs": [completed, waiting],
                    },
                    {
                        "name": "deploy",
                        "status": "created",
                        "verdict": None,
                        "jobs": [make_job()],
                    },
                ],
            }
            create_process = AsyncMock(
                return_value=FakeProcess("Question?\nSTATUS: WAIT - Question?\n", 0)
            )
            self.addCleanup(app.RUN_PROCS.pop, run["id"], None)

            with (
                patch("app.asyncio.create_subprocess_exec", create_process),
                patch("app.save_run"),
            ):
                await app.execute_run(run)

        self.assertEqual(create_process.await_count, 1)
        self.assertEqual(run["status"], app.WAITING)
        self.assertEqual(run["stages"][0]["status"], app.WAITING)
        self.assertEqual(run["stages"][1]["status"], "created")
        self.assertEqual(completed["status"], "success")

    async def test_approve_requires_external_user_message(self):
        job = make_job()
        job.update(
            status=app.WAITING,
            verdict=app.WAIT,
        )
        stage = {"name": "review", "status": app.WAITING, "verdict": app.WAIT, "jobs": [job]}
        run = {
            "id": "test-answer",
            "status": app.WAITING,
            "finished_at": None,
            "stages": [stage],
        }

        with patch.dict(app.RUNS, {run["id"]: run}):
            with self.assertRaises(app.HTTPException) as error:
                await app.approve_waiting_job(run["id"], 0, 0)

        self.assertEqual(error.exception.status_code, 409)
