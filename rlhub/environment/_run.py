import uuid

from temporalio import client

from rlhub.agent._base import BaseRunner as BaseAgentRunner
from rlhub.common.model import Event
from rlhub.environment._base import BaseRunner


class RemoteRunner(BaseRunner):
    """
    Environment runner sub class that runs environment activities from anywhere.
    """

    def __init__(self, client: client.Client, agent_task_queue: str = "agent"):
        """
        Initialize the environment runner with a task queue.
        """
        super().__init__()
        self._client = client
        self._agent_task_queue = agent_task_queue
        self._env_id = str(uuid.uuid4())

    @property
    def agent_task_queue(self) -> str:
        """
        The task queue to use for the agent.
        """
        return self._agent_task_queue

    async def run_impl(self, input_data: BaseRunner.RunParams) -> str:
        """
        Run the environment with an agent in Temporal Cloud.
        """

        # get initial state
        state = input_data.initial_state
        if state is None:
            state = self.init()

        # run for an episode
        event = Event(
            state=state,
            action=None,
            reward=None,
            done=False,
        )
        while not event.done:
            # plan action with agent
            action = await self._client.execute_update_with_start_workflow(
                BaseAgentRunner.serve_policy,
                state,
                start_workflow_operation=client.WithStartWorkflowOperation(
                    BaseAgentRunner.run,
                    id=self.agent_task_queue + self._env_id,
                    task_queue=self.agent_task_queue,
                ),
                id=self.agent_task_queue + self._env_id,
            )

            # execute action in environment
            result = self.act(BaseRunner.ActParams(action=action, state=state))
            state = result.state
            event.reward = result.reward
            event.done = result.done
            event.action = action

            # upload event to agent
            await self._client.start_workflow(
                BaseAgentRunner.run,
                id=self.agent_task_queue + self._env_id,
                task_queue=self.agent_task_queue,
                start_signal=BaseAgentRunner.upload_history,
                signal_args=[event],
            )


# Should also have LocalEnvironmentRunner that runs environment with local activities
# to guarantee that the environment is run in the same process (would be easier for 1to1 or resource management)

# Also a LocalLocalEnvironmentRunner that runs the environment with local activities
# and uses local activities for the agent
# maybe it takes an agent as an input
# and calls its implementation

# Also some that use nexus for later on?
