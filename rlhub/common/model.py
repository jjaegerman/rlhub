from dataclasses import dataclass

type Action = dict
type Reward = float
type Done = bool
type State = dict


@dataclass
class Event:
    """
    Event class to represent an event in the environment.
    """

    state: State
    action: Action
    reward: Reward
    done: Done
