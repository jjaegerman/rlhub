from abc import ABC, abstractmethod
from typing import Any

from temporalio import activity, workflow


@workflow.defn(name="RunFactory")
class BaseFactory(ABC):
    """
    Interface for a workflow that manages a factory endpoint.
    This encapsulates data sampling, evicting, model training and
    policy distribution.
    """

    def __init__(self) -> None:
        """
        Initialize the factory runner with a task queue.
        """
        self._task_queue = "factory"

    @property
    def task_queue(self) -> str:
        """
        The task queue to use for the factory.
        """
        return self._task_queue

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
    async def upload_batch(self, batch: Any) -> None:
        """
        Receive a batch of data from agent.
        """
        await self.upload_batch_impl(batch)

    @abstractmethod
    async def upload_batch_impl(self, batch: Any) -> None:
        """
        Receive a batch of data from agent.
        """
        pass

    @activity.defn(name="SampleBatch")
    async def sample_batch(
        self,
        batch: Any,
    ) -> Any:
        """
        Sample from the batch of data.
        """
        return await self.sample_batch_impl(batch)

    @abstractmethod
    async def sample_batch_impl(
        self,
        batch: Any,
    ) -> Any:
        """
        Sample from the batch of data.
        """
        pass

    @activity.defn(name="EvictBuffer")
    async def evict_buffer(
        self,
        buffer: Any,
    ) -> Any:
        """
        Evict from the replay buffer.
        """
        return await self.evict_buffer_impl(buffer)

    @abstractmethod
    async def evict_buffer_impl(
        self,
        buffer: Any,
    ) -> Any:
        """
        Evict from the replay buffer.
        """
        pass

    @abstractmethod
    def trigger_training(self) -> bool:
        """
        Trigger training based on a condition.
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

    @abstractmethod
    def trigger_redistribution(self) -> bool:
        """
        Trigger redistribution of the policy.
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
