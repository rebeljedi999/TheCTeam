from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

class UnrealFPS(py_environment.PyEnvironment):
    def __init__(self):
        # define action that will be recieved
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=7, name='play')
        # there are a total of 8 actions that can be triggered

        # observation will look like the following:
        #   [can_see_enemy(0, 1), visual_sensor(some sort of array), health(0 - 100), time(0 - 120), win(0, 1)]
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, 5), dtype='object', minimum=[0, None, 0, 0, 0], maximum=[1, None, 100, 120, 1], name='sensors')
        self._state = [0, None, 0, 120, 0]
        self._episode_ended = False
        self.reward = 0
        self.action = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # set state at beginning of a game
        self._state = [0, None, 0, 120, 0]
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=object))

    def _step(self, act):
        if self._episode_ended:
            return self.reset()

        # if agent completes all objectives before time runs out, end episode and give reward
        if self._state[4] == 1:
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=object), 10)

        # if time runs out, end episode and negative reward
        if self._state[3] <= 0:
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=object), -10)

        # if agent is still alive, give reward
        if self._state[2] > 0:
            return ts.transistion(np.array([self._state], dtype=object), reward=reward, discount=1.0)

        # otherwise agent is dead, terminate with negative reward
        else:
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=object), -10)

