from abc import ABC, abstractmethod
from typing import Any, Sequence

from temporalio import activity, workflow

from rlhub.common.model import Action, Event, State


@workflow.defn(name="ManageAgent")
class BaseRunner(ABC):
    """
    Interface for a workflow that manages an agent endpoint.
    This encapsulates data collection and policy maintenance.
    """

    def __init__(self, task_queue: str = "agent") -> None:
        """
        Initialize the agent runner with a task queue.
        """
        self._task_queue = task_queue

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
        return self.run_impl(input_data)

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
        return self.upload_history_impl(history)

    @abstractmethod
    async def upload_history_impl(
        self,
        history: Sequence[Event],
    ) -> Any:
        """
        Upload the history to the agent manager.
        """
        pass

    @activity.defn(name="SwapPolicy")
    async def swap_policy(
        self,
        new_policy: Any,
    ) -> Any:
        return self.swap_policy_impl(new_policy)

    @abstractmethod
    async def swap_policy_impl(
        self,
        new_policy: Any,
    ) -> Any:
        """
        Swap the policy in the agent manager.
        """
        pass

    # TODO: implement update approach ocne supported for better observability
    @activity.defn(name="ExecutePolicy")
    async def execute_policy(self, state: State) -> Action:
        """
        Determine the next action to take in the environment.
        """
        return self.execute_policy_impl(state)

    @abstractmethod
    async def execute_policy_impl(self, state: State) -> Action:
        """
        Determine the next state to take in the environment.
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
