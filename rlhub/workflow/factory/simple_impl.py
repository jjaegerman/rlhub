from abc import abstractmethod
from typing import Any, Dict, List

from temporalio import workflow

from rlhub.common.model import Event
from rlhub.workflow.factory._base import (
    BaseFactory,
    BaseFactoryActivities,
    PolicyDistribution,
)


class SimpleFactoryActivities(BaseFactoryActivities):
    async def sample_batch_impl(
        self,
        key: str,
    ) -> None:
        """
        Sample from the batch of data.
        And evict from Replay Buffer.
        """
        # asyncio doesn't preempt so this is fine
        SimpleFactory.replay_buffer.extend(SimpleFactory.batch_stage[key])
        SimpleFactory.new_data = True

    async def redistribute_policy_impl(
        self,
    ) -> List[Event]:
        """
        Redistribute the policy.
        """
        SimpleFactory.distribution.asset = SimpleFactory._latest
        SimpleFactory.distribution.version += 1

    async def train_model_impl(
        self,
    ) -> Any:
        """
        Train the model.
        """
        self.do_train()
        SimpleFactory.training = False

    @abstractmethod
    async def do_train(self):
        pass


class SimpleFactory(BaseFactory):
    batch_stage: Dict[str, List[Event]] = dict()
    replay_buffer: List[Event] = list()
    _latest: Any
    distribution: PolicyDistribution = PolicyDistribution(version=0)
    training = False
    new_data = False

    async def run_impl(self, input_data: Any) -> Any:
        """
        Run the factory manager with the given arguments.
        """
        await workflow.info().is_continue_as_new_suggested()
        await workflow.wait_condition(workflow.all_handlers_finished)
        return workflow.continue_as_new(input_data)

    async def upload_batch_impl(self, key: str) -> None:
        """
        Receive a batch of data from agent.
        """
        await workflow.execute_activity(BaseFactoryActivities.sample_batch, key)
        del SimpleFactory.batch_stage[key]
        if self.trigger_training():
            SimpleFactory.training = True
            await workflow.execute_activity(
                BaseFactoryActivities.train_model,
            )
            if self.trigger_redistribution():
                await workflow.execute_activity(
                    BaseFactoryActivities.redistribute_policy
                )

    def trigger_training(self) -> bool:
        """
        Trigger training based on a condition.
        """
        return SimpleFactory.training and SimpleFactory.new_data

    def trigger_redistribution(self) -> bool:
        """
        Trigger redistribution of the policy.
        """
        return True
