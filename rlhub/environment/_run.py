from datetime import timedelta

from temporalio import workflow

from rlhub.common.model import Event
from rlhub.environment._base import Base


class RemoteActivityRunner(Base):
    """
    Environment runner sub class that runs environment activities from anywhere.
    """

    def __init__(
        self, task_queue: str = "environment", agent_task_queue: str = "agent"
    ):
        """
        Initialize the environment runner with a task queue.
        """
        super().__init__(task_queue)
        self._agent_task_queue = agent_task_queue

    @property
    def agent_task_queue(self) -> str:
        """
        The task queue to use for the agent.
        """
        return self._agent_task_queue

    async def run_impl(self, input_data: Base.RunParams) -> str:
        """
        Run the environment with an agent in Temporal Cloud.
        """

        # get initial state
        state = input_data.initial_state
        if state is None:
            state = await workflow.execute_activity(
                self.init,
                task_queue=self.task_queue,
                start_to_close_timeout=timedelta(seconds=10),
            )

        # run for an episode
        event = Event(
            state=state,
            action=None,
            reward=None,
            done=False,
        )
        while not event.done:
            # plan action with agent
            action = await workflow.execute_activity(
                self.plan,
                Base.PlanParams(state=state, prev_event=event),
                task_queue=self.agent_task_queue,
                start_to_close_timeout=timedelta(seconds=10),
            )

            # execute action in environment
            result = await workflow.execute_activity(
                self.act,
                Base.ActParams(action=action, state=state),
                task_queue=self.task_queue,
                start_to_close_timeout=timedelta(seconds=10),
            )
            state = result.state
            event.reward = result.reward
            event.done = result.done
            event.action = action

        # next episode
        workflow.continue_as_new(Base.RunParams())


# Should also have LocalEnvironmentRunner that runs environment with local activities
# to guarantee that the environment is run in the same process (would be easier for 1to1 or resource management)

# Also a LocalLocalEnvironmentRunner that runs the environment with local activities
# and uses local activities for the agent
# maybe it takes an agent as an input
# and calls its implementation

# Also some that use nexus for later on?
