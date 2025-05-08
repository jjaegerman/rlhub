from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from temporalio import activity, workflow

from rlhub.common.model import Action, Done, Reward, State


@workflow.defn(name="RunEnvironment")
class BaseRunner(ABC):
    """
    Interface for a workflow that runs an environment.
    This encapsulates environment interfacing and execution
    strategy.
    """

    def __init__(self, task_queue: str = "environment-runner") -> None:
        """
        Initialize the environment runner with a task queue.
        """
        self._task_queue = task_queue

    @property
    def task_queue(self) -> str:
        """
        The task queue to use for the environment.
        """
        return self._task_queue

    @dataclass
    class RunParams:
        """
        Parameters for the run method.
        """

        initial_state: Optional[dict] = None

    @workflow.run
    async def run(self, input_data: RunParams) -> Any:
        """
        Run the environment with the given arguments.
        """
        return self.run_impl(input_data)

    @abstractmethod
    async def run_impl(self, input_data: RunParams) -> Any:
        """
        Run the environment with the given arguments.
        """
        pass

    @activity.defn(name="Init")
    async def init(self) -> State:
        """
        Initialize a new environment instance and
        return the initial state.
        """
        return self.init_impl()

    @abstractmethod
    async def init_impl(self) -> State:
        """
        Initialize a new environment instance and
        return the initial state.
        """
        pass

    @dataclass
    class ActParams:
        """
        Parameters for the act method.
        """

        action: Action
        state: Optional[State] = None

    @dataclass
    class ActResult:
        """
        Result of the act method.
        """

        state: State
        reward: Reward
        done: Done

    @activity.defn(name="Act")
    async def act(self, action: ActParams) -> ActResult:
        """
        Perform an action in the environment and
        observe resulting state and reward.
        """
        return self.act_impl(action)

    @abstractmethod
    async def act_impl(self, action: ActParams) -> ActResult:
        """
        Perform an action in the environment and
        observe resulting state and reward.
        """
        pass
