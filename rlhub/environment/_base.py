from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from rlhub.common.model import Action, Done, Reward, State


class BaseRunner(ABC):
    """
    Interface for a class that runs an environment.
    This encapsulates environment interfacing and execution
    strategy.
    """

    @dataclass
    class RunParams:
        """
        Parameters for the run method.
        """

        initial_state: Optional[dict] = None

    @abstractmethod
    async def run(self, input_data: RunParams) -> Any:
        """
        Run the environment with the given arguments.
        """
        pass

    @abstractmethod
    async def init(self) -> State:
        """
        Initialize a new environment instance and
        returns a copy of the initial state.
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

    async def act(self, action: ActParams) -> ActResult:
        """
        Perform an action in the environment and
        observe resulting state and reward.
        """
        pass
