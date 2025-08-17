from abc import ABC, abstractmethod
from typing import Any, List

from temporalio import activity, workflow

from rlhub.common.model import Event


class BaseFactoryActivities(ABC):
    @activity.defn(name="SampleBatch")
    async def sample_batch(
        self,
        batch: List[Event],
    ) -> Any:
        """
        Sample from the batch of data.
        And evict from Replay Buffer.
        """
        return await self.sample_batch_impl(batch)

    @abstractmethod
    async def sample_batch_impl(
        self,
        batch: List[Event],
    ) -> Any:
        """
        Sample from the batch of data.
        And evict from Replay Buffer.
        """
        pass

    @activity.defn(name="RedistributePolicy")
    async def redistribute_policy(
        self,
        policy: Any,
    ) -> Any:
        """
        Redistribute the policy.
        """
        return await self.redistribute_policy_impl(policy)

    @abstractmethod
    async def redistribute_policy_impl(
        self,
        policy: Any,
    ) -> Any:
        """
        Redistribute the policy.
        """
        pass

    @activity.defn(name="TrainModel")
    async def train_model(
        self,
        model: Any,
    ) -> Any:
        """
        Train the model.
        """
        return await self.train_model_impl(model)

    @abstractmethod
    async def train_model_impl(
        self,
        model: Any,
    ) -> Any:
        """
        Train the model.
        """
        pass


@workflow.defn(name="RunFactory")
class BaseFactory(ABC):
    """
    Interface for a workflow that manages a factory endpoint.
    This encapsulates data sampling, evicting, model training and
    policy distribution.
    """

    @workflow.run
    async def run(self, input_data: Any) -> Any:
        """
        Run the factory manager with the given arguments.
        """
        return await self.run_impl(input_data)

    @abstractmethod
    async def run_impl(self, input_data: Any) -> Any:
        """
        Run the factory manager with the given arguments.
        """
        pass

    @workflow.signal(name="UploadBatch")
    async def upload_batch(self, batch: List[Event]) -> None:
        """
        Receive a batch of data from agent.
        """
        await self.upload_batch_impl(batch)

    @abstractmethod
    async def upload_batch_impl(self, batch: List[Event]) -> None:
        """
        Receive a batch of data from agent.
        """
        pass

    @abstractmethod
    def trigger_training(self) -> bool:
        """
        Trigger training based on a condition.
        """
        pass

    @abstractmethod
    def trigger_redistribution(self) -> bool:
        """
        Trigger redistribution of the policy.
        """
        pass
