from abc import ABC, abstractmethod

from rlhub.common.model import Action, Event, State


class BaseAgent(ABC):
    """
    Interface for an agent endpoint.
    This encapsulates data staging and policy serving.
    """

    @abstractmethod
    async def serve_policy(self, state: State) -> Action:
        """
        Serve the policy to the agent manager.
        """
        pass

    # TODO: This will need to include output probability of action taken too
    @abstractmethod
    async def record_event(self, event: Event) -> None:
        """
        Record an event in the agent manager.
        """
        pass


class BasePolicy(ABC):
    """
    Interface for a singleton policy.
    This encapsulates policy management and access.
    """

    @abstractmethod
    def execute(self, state: State) -> Action:
        """
        Execute the policy on the given state.
        """
        pass
