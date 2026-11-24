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
