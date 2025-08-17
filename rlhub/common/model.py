from dataclasses import dataclass
from typing import Any

type Action = Any
type Reward = float
type Done = bool
type State = Any


@dataclass
class Event:
    """
    Event class to represent an event in the environment.
    """

    state: State
    action: Action
    reward: Reward
    done: Done
