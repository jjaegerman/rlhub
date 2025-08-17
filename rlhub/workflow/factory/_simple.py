from typing import Any, List

from rlhub.common.model import Event
from rlhub.workflow.factory._base import BaseFactory, BaseFactoryActivities


class SimpleFactoryActivities(BaseFactoryActivities):
    async def sample_batch_impl(
        self,
        batch: List[Event],
    ) -> List[Event]:
        """
        Sample from the batch of data.
        And evict from Replay Buffer.
        """
        pass

    async def redistribute_policy_impl(
        self,
        policy: Any,
    ) -> List[Event]:
        """
        Redistribute the policy.
        """
        pass

    async def train_model_impl(
        self,
        model: Any,
    ) -> Any:
        """
        Train the model.
        """
        pass


class SimpleFactory(BaseFactory):
    async def run_impl(self, input_data: Any) -> Any:
        """
        Run the factory manager with the given arguments.
        """
        pass

    async def upload_batch_impl(self, batch: List[Event]) -> None:
        """
        Receive a batch of data from agent.
        """
        pass

    def trigger_training(self) -> bool:
        """
        Trigger training based on a condition.
        """
        pass

    def trigger_redistribution(self) -> bool:
        """
        Trigger redistribution of the policy.
        """
        pass
