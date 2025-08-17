import uuid

from temporalio import client

from rlhub.common.model import Event
from rlhub.workflow.agent._remote import RemoteBaseAgent
from rlhub.workflow.agent.remote_impl import RemoteAgent
from rlhub.workflow.environment._base import BaseRunner


class RemoteRunner(BaseRunner):
    """
    Environment runner that uses a remote agent.
    """

    def __init__(self, client: client.Client, agent_task_queue: str = "agent"):
        """
        Initialize the environment runner with a task queue.
        """
        super().__init__()
        self._client = client
        self._agent_task_queue = agent_task_queue
        self._agent_workflow_id = "agent"
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
                RemoteBaseAgent.serve_policy,
                state,
                start_workflow_operation=client.WithStartWorkflowOperation(
                    RemoteBaseAgent.run,
                    id=self._agent_workflow_id,
                    task_queue=self.agent_task_queue,
                ),
            )

            # execute action in environment
            result = self.act(BaseRunner.ActParams(action=action, state=state))
            state = result.state
            event.reward = result.reward
            event.done = result.done
            event.action = action

            # upload event to agent
            event_key = str(uuid.uuid4())
            RemoteAgent.eventStage[event_key] = event
            await self._client.get_workflow_handle(self._agent_workflow_id).signal(
                RemoteBaseAgent.record_event, event_key
            )


# Should also have LocalEnvironmentRunner that runs environment with function invocation
