from abc import abstractmethod
from typing import Any

from temporalio import workflow

from rlhub.agent._base import BaseAgent, BasePolicy
from rlhub.common.model import Action, Event, State


@workflow.defn(name="ManagePolicy")
class RemoteBasePolicy(BasePolicy):
    """
    Interface for a workflow that manages a singleton policy.
    """

    @workflow.run
    async def run(self):
        """
        Initialization and management logic
        """
        return await self.run_impl()

    @abstractmethod
    async def run_impl(self):
        pass


@workflow.defn(name="RunAgent")
class RemoteBaseAgent(BaseAgent):
    """
    Interface for a workflow that manages an agent endpoint.
    This encapsulates data staging and policy serving.
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
