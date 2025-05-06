from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from temporalio import activity, workflow

from rlhub.common.model import Action, Done, Event, Reward, State


@workflow.defn(name="RunEnvironment")
class Base(ABC):
    """
    Interface for a workflow that runs an environment.
    This encapsulates environment interfacing and execution
    strategy.
    """

    def __init__(self, task_queue: str = "environment"):
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
    async def run(self, input_data: RunParams) -> str:
        """
        Run the environment with the given arguments.
        """
        return self.run_impl(input_data)

    @abstractmethod
    async def run_impl(self, input_data: RunParams) -> str:
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

    @dataclass
    class PlanParams:
        """
        Parameters for the plan method.
        """

        state: State
        prev_event: Optional[Event] = None

    @activity.defn(name="Plan")
    async def plan(self, state: State) -> Action:
        """
        Determine the next action to take in the environment.
        And optionally pass the previous event for learning.
        """
        return self.plan_impl(state)

    @abstractmethod
    async def plan_impl(self, state: State) -> Action:
        """
        Determine the next state to take in the environment.
        And optionally pass the previous event for learning.
        """
        pass
