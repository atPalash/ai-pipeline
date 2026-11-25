from pathlib import Path
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
