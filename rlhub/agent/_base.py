from abc import ABC, abstractmethod
from typing import Any, Sequence

from temporalio import activity, workflow

from rlhub.common.model import Action, Event, State


@workflow.defn(name="ManageAgent")
class BaseManager(ABC):
    """
    Interface for a workflow that manages an agent endpoint.
    This encapsulates data collection and policy maintenance.
    """

    def __init__(self, task_queue: str = "agent-supervisor") -> None:
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


@workflow.defn(name="Plan")
class BaseRunner(ABC):
    """
    Interface for a workflow that serves an agent endpoint.
    This encapsulates planning and policy computation.
    """

    def __init__(self, task_queue: str = "agent-runner"):
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
    async def run(self, state: State) -> Action:
        """
        Determine the next action to take in the environment.
        """
        return self.run_impl(state)

    @abstractmethod
    async def run_impl(self, state: State) -> Action:
        """
        Determine the next action to take in the environment.
        """
        pass

    @activity.defn(name="Plan")
    async def plan(self, state: State) -> Action:
        """
        Determine the next action to take in the environment.
        """
        return self.plan_impl(state)

    @abstractmethod
    async def plan_impl(self, state: State) -> Action:
        """
        Determine the next state to take in the environment.
        """
        pass
