import asyncio
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any

from temporalio import workflow

from rlhub.agent._base import BaseAgent
from rlhub.common.model import Action, Event, State


class Policy(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        """
        The version of the policy.
        """
        pass

    @property
    @abstractmethod
    def updating(self) -> bool:
        """
        Whether the policy is currently being updated.
        """
        pass

    @abstractmethod
    def execute(self, state: State) -> Action:
        """
        Execute the policy on the given state.
        """
        pass

    @abstractmethod
    async def update(self) -> None:
        """
        Update the policy with the latest version.
        Set self.version to the new version use locking
        if there is awaiting.
        """
        pass


class CommonAgent(BaseAgent):
    """
    CommonAgent is an abstract class for RL agents that provides a common
    implementation of the main agent workflow and message handlers.
    Implements history batching and basic policy serving.
    """

    @property
    @abstractmethod
    def policy(self) -> Policy:
        """
        The policy used by the agent.
        """
        pass

    def init_impl(self, input_data: Any) -> None:
        self.batch_size = input_data.get("batch_size", 100)
        self.execute_timeout = input_data.get("execute_timeout", 60)
        self.poll_policy_interval = input_data.get("poll_policy_interval", 60)

        self.history_buffer = list()
        self.history_buffer_lock = asyncio.Lock()
        self.latest_policy_version = None

    async def run_impl(self, input_data: Any) -> Any:
        while not workflow.info().is_continue_as_new_suggested():
            self.latest_policy_version = await workflow.execute_activity(
                self.poll_policy,
                task_queue=self.task_queue,
                start_to_close_timeout=timedelta(seconds=15),
            )
            await asyncio.sleep(self.poll_policy_interval)
        await workflow.wait_condition(workflow.all_handlers_finished)
        return workflow.continue_as_new(
            input_data,
            task_queue=self.task_queue,
        )

    async def record_event_impl(self, event: Event) -> None:
        self.history_buffer.append(event)

        if len(self.history_buffer) >= self.batch_size:
            # asyncio does not preempt - no need to synchronize
            slice = self.history_buffer[: self.batch_size]
            self.history_buffer = self.history_buffer[self.batch_size :]
            await workflow.execute_activity(
                self.upload_history,
                slice,
                task_queue=self.task_queue,
                start_to_close_timeout=timedelta(seconds=self.execute_timeout),
            )

    async def serve_policy_impl(self, state: State) -> Action:
        return await workflow.execute_activity(
            self.execute_policy,
            state,
            task_queue=self.task_queue,
            start_to_close_timeout=timedelta(seconds=self.execute_timeout),
        )

    async def execute_policy_impl(self, state: State) -> Action:
        if (
            self.policy.version != self.latest_policy_version
            and not self.policy.updating
        ):
            asyncio.create_task(self.policy.update())
        return self.policy.execute(state)
