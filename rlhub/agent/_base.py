from abc import ABC, abstractmethod
from typing import Any, Sequence

from temporalio import activity, workflow

from rlhub.common.model import Action, Event, State


@workflow.defn(name="RunAgent")
class BaseAgent(ABC):
    """
    Interface for a workflow that manages an agent endpoint.
    This encapsulates data collection and policy maintenance.
    """

    @workflow.init
    def __init__(self, input_data: Any) -> None:
        """
        Initialize the agent runner with a task queue.
        """
        self._task_queue = "agent"
        self.init_impl(input_data)

    @abstractmethod
    def init_impl(self, input_data: Any) -> None:
        """
        Initialize the agent runner.
        """
        pass

    @property
    def task_queue(self) -> str:
        """
        The task queue to use for the agent.
        """
        return self._task_queue

    @workflow.run
    async def run(self, input_data: Any) -> Any:
        """
        Run the agent manager with the given arguments.
        """
        return await self.run_impl(input_data)

    @abstractmethod
    async def run_impl(self, input_data: Any) -> Any:
        """
        Run the agent manager with the given arguments.
        """
        pass

    @activity.defn(name="UploadHistory")
    async def upload_history(
        self,
        history: Sequence[Event],
    ) -> Any:
        return await self.upload_history_impl(history)

    @abstractmethod
    async def upload_history_impl(
        self,
        history: Sequence[Event],
    ) -> Any:
        """
        Upload the history to the agent manager.
        """
        pass

    @activity.defn(name="PollPolicy")
    async def poll_policy(self) -> str:
        return self.poll_policy_impl()

    @abstractmethod
    async def poll_policy_impl(self) -> str:
        """
        Poll the agent manager for a new policy version.
        """
        pass

    @activity.defn(name="ExecutePolicy")
    async def execute_policy(self, state: State) -> Action:
        """
        Determine the next action to take in the environment.
        """
        return await self.execute_policy_impl(state)

    @abstractmethod
    async def execute_policy_impl(self, state: State) -> Action:
        """
        Determine the next state to take in the environment.
        """
        pass

    @workflow.update(name="ServePolicy")
    async def serve_policy(self, state: State) -> Action:
        """
        Serve the policy to the agent manager.
        """
        return await self.serve_policy_impl(state)

    @abstractmethod
    async def serve_policy_impl(self, state: State) -> Action:
        """
        Serve the policy to the agent manager.
        """
        pass

    # TODO: This will need to include output probability of action taken too
    @workflow.signal(name="RecordEvent")
    async def record_event(self, event: Event) -> None:
        """
        Record an event in the agent manager.
        """
        await self.record_event_impl(event)

    @abstractmethod
    async def record_event_impl(self, event: Event) -> None:
        """
        Record an event in the agent manager.
        """
        pass
