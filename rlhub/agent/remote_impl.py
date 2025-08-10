    @activity.defn(name="UploadHistory")
    async def upload_history(
        self,
        history: Sequence[Event],
    ) -> Any:
        return await self.upload_history_impl(history)

    @abstractmethod
    async def upload_history_impl(
        self,
        history: Sequence[Event],
    ) -> Any:
        """
        Upload the history to the agent manager.
        """
        pass

    @activity.defn(name="ExecutePolicy")
    async def execute_policy(self, state: State) -> Action:
        """
        Determine the next action to take in the environment.
        """
        return await self.execute_policy_impl(state)

    @abstractmethod
    async def execute_policy_impl(self, state: State) -> Action:
        """
        Determine the next state to take in the environment.
        """
        pass