import gym
from math import inf
import numpy as np
from collections import deque,defaultdict
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
# other than the rewardwrapper which is customly made, most wrappers here were copied from gym.retro's own wrappers
# with slight adjustments and improvments
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps=0
        self.last_info=dict([])
        self.last_scaled_prog=0
        self.before_last_info=dict([])
        #stores the max x coordinates reached for each level
        self.max_x=0
        # end of level coordinates for sonic, used in the rewards wrapper
        self.level_max_x = {
            (0,0) : 0x2560,
            (0,1) : 0x1F60,
            (0,2) : 0x292A,

            (2,0) : 0x1860,
            (2,1) : 0x1860,
            (2,2) : 0x1720,

            (4,0) : 0x2360,
            (4,1) : 0x2960,
            (4,2) : 0x2B83,

            (1,0) : 0x1A50,
            (1,1) : 0x1150,
            (1,2) : 0x1CC4,

            (3,0) : 0x2060,
            (3,1) : 0x2060,
            (3,2) : 0x1F48,

            (5,0) : 0x2260,
            (5,1) : 0x1EE0,
            (5,2) : inf
        }
    def reset(self, **kwargs):
        self.last_info = dict([])
        self.last_scale_prog=0
        self.before_last_info=dict([])
        self.max_x=0
        self.steps=0
        return self.env.reset(**kwargs)
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew=0 # ignore default rew and calculate one based on info and last_info
        won=False
        if len(self.last_info) > 0 and info["screen_x_end"]!=0:
            # no reward estimation for first step
            lvl = (info["zone"],info["act"])
            level_max_x=self.level_max_x[lvl] # maximum possible coordinates for this level
            scaled_prog = info["x"]/level_max_x # scaled prog
            d_max_x= level_max_x-info["x"]
            if scaled_prog>0.95 or info["x"]>=30000:
                # close enough to the end to be considered a win
                won = True
                done = True
                #calculate some bonus for winning according to how fast it was done
                bonus= 5000-self.steps
                if bonus < 100:
                    bonus=100
                rew+=bonus
            if info["x"] > self.last_info["x"]:
                #add reward for positive changes in x scaled by level's length
                if lvl != (5,2):
                    rew+=(scaled_prog-self.last_scaled_prog)*5000
                else:
                    rew+=1

            if self.before_last_info:
                d_x=info["x"]-self.before_last_info["x"]
                #motivate running
                if d_x>2:
                    rew+=0.5
            if info["x"] < self.last_info["x"]:
                pass
                #rew-=1
            if info["x"] == self.last_info["x"] and info["y"] == self.last_info["y"]:
                #motivate movement
                pass
                #rew-=0.5
            if info["lives"] < self.last_info["lives"]:
                #rew-=2
                self.last_life_lost=self.steps
            d_score = info["score"]-self.last_info["score"]
            if d_score > 0:
                rew+=d_score*0.001
            if info["rings"] > self.last_info["rings"]:
                rew+=0.2
            if self.last_info["rings"] > 0 and info["rings"] == 0:
                # if we were attacked and lost all rings
                pass
                #rew-=1
            if info["x"]>self.max_x:
                #if this point was the farthest we've reached in this level (in these t timesteps)
                rew+=3
                self.max_x = info["x"]
            self.last_scaled_prog=scaled_prog
        self.steps+=1
        self.last_info = info
        self.before_last_info = self.last_info
        return won, obs, rew, done, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], ['UP']] #adding UP
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

