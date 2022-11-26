from gym import Env
from numpy import ndarray


class TrainStats:
    pass


class Agent:
    def __init__(self) -> None:
        pass

    def act(self, state: ndarray) -> ndarray:
        raise NotImplementedError()

    def step(self, state: ndarray) -> ndarray:
        raise NotImplementedError()

    def learn(
        self,
        state: ndarray,
        action: int,
        reward: float,
        state_: ndarray,
        terminated: bool,
        done: bool,
    ) -> TrainStats:
        pass

    def save(self, path: str) -> None:
        pass

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


class RandomAgent(Agent):
    """An example Random Agent"""

    def __init__(self, env: Env) -> None:
        super().__init__()
        self._env = env

    def act(self, state: ndarray) -> ndarray:
        return self._env.action_space.sample()

    def step(self, state: ndarray) -> ndarray:
        return self.act(state)
