import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import torch
from temporalio import activity, workflow

from rlhub.common.model import Action, Event, State
from rlhub.workflow.agent._remote import RemoteBaseAgent, RemoteBasePolicy
from rlhub.workflow.factory._base import BaseFactory


# can reorganize later but this is a pytorch policy (torchscript) with
# a dedicated update workflow polled from local storage
# assumes model .pt torchscript files are stored in dir with a filename
# that increases lexicographically with version
# the workflow should run on a machine-specific task queue
class RemotePolicy(RemoteBasePolicy):
    workflow_id = "policy_poller"
    _policy = None
    _policy_ready = asyncio.Event()
    _version = None
    _lock = asyncio.Lock()

    @activity.defn(name="DoPoll")
    def do_poll(self, poll_dir) -> None:
        search_directory = Path(poll_dir)
        file_pattern = "*.pt"

        files = search_directory.glob(file_pattern)
        latest = max(files, lambda path: path.stem)

        # asyncio does not preempt - no need to synchronize
        if RemotePolicy._version is None:
            RemotePolicy._version = latest
            RemotePolicy._policy = torch.jit.load(self._poll_path)
            RemotePolicy._policy_ready.set()

    @property
    def task_queue(self) -> str:
        return self._task_queue

    @property
    def poll_dir(self) -> str:
        return self._poll_dir

    @property
    def poll_interval(self) -> timedelta:
        return self.poll_interval

    @workflow.init
    def __init__(self, input_data: Dict[str, Any]) -> None:
        self._poll_dir = input_data.get("poll_path")
        self._poll_interval = input_data.get("poll_interval")
        self._task_queue = input_data.get("task_queue")

    async def run_impl(self, input_data: Dict[str, Any]):
        while not workflow.info().is_continue_as_new_suggested():
            await workflow.execute_activity(
                RemotePolicy.do_poll, self.poll_dir, task_queue=self.task_queue
            )
            workflow.sleep(self.poll_interval)
        return workflow.continue_as_new(
            input_data,
            task_queue=self.task_queue,
        )

    @staticmethod
    async def execute(state: State):
        # probably a better way to do this
        # maybe we can have the agent worker wait
        # for this event to start itself so this would
        # never happen
        if not RemotePolicy._policy:
            await RemotePolicy._policy_ready.wait()

        return Action(RemotePolicy._policy(state))


@activity.defn(name="ExecutePolicy")
async def execute_remote_policy(state: State) -> Action:
    return await RemotePolicy.execute(state)


# ditto this is a simple agent, no request batching, simple history batching
# retrieves samples from local storage
# doesn't do any sticky sessions for GAE or episode-batch normalization of A or other preprocessing techniques
class RemoteAgent(RemoteBaseAgent):
    @workflow.init
    def __init__(self, input_data: Dict[str, Any]) -> None:
        self._task_queue = "agent"
        self.factory_id = input_data.get("factory_id", "factory")

        self.history_batch_size = input_data.get("batch_size", 100)
        self.execute_timeout = input_data.get("execute_timeout", 60)
        self.poll_policy_interval = input_data.get("poll_policy_interval", 60)

        self.history_buffer = list()
        self.history_buffer_lock = asyncio.Lock()
        self.latest_policy_version = None

    @property
    def task_queue(self) -> str:
        return self._task_queue

    async def run_impl(self, input_data: Dict[str, Any]) -> Any:
        await workflow.info().is_continue_as_new_suggested()
        await workflow.wait_condition(workflow.all_handlers_finished)
        return workflow.continue_as_new(
            input_data,
            task_queue=self.task_queue,
        )

    async def serve_policy_impl(self, state: State) -> Action:
        return await workflow.execute_activity(
            execute_remote_policy,
            state,
            task_queue=self.task_queue,
            start_to_close_timeout=timedelta(seconds=self.execute_timeout),
        )

    async def record_event_impl(self, event: Event) -> None:
        self.history_buffer.append(event)

        if len(self.history_buffer) >= self.history_batch_size:
            # asyncio does not preempt - no need to synchronize
            slice = self.history_buffer[: self.history_batch_size]
            self.history_buffer = self.history_buffer[self.history_batch_size :]
            await workflow.get_external_workflow_handle(self.factory_id).signal(
                BaseFactory.upload_batch, slice
            )
