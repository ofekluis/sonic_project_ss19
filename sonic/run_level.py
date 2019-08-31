import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import wrappers
import gym
import retro
import effnet
import os
frames_stack=4
ACTION_SIZE=8
learning_rate = 5e-5
target_model = effnet.Effnet(input_shape=(128,128,frames_stack),nb_classes=ACTION_SIZE, info=11)
if os.path.isfile("sonic_target_model.h5"):
    target_model.load_weights("sonic_target_model.h5")
else:
    print("No weights found, exiting")
    sys.exit(1)
target_model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
target_model.trainable = False
game = "SonicTheHedgehog-Genesis"
if len(sys.argv)>1:
    print(sys.argv[1])
    state=str(sys.argv[1])
else:
    # if level not given use first level
    state="GreenHillZone.Act1"
env = retro.make(game, state,scenario="scenario.json", record="logs/")
env = wrappers.WarpFrame(env, 128, 128, grayscale=True)
env = wrappers.FrameStack(env,frames_stack)
env = wrappers.SonicDiscretizer(env) # Discretize the environment for q learning
env = wrappers.RewardWrapper(env) # custom reward calculation
obs = env.reset() #game start
obs = np.array(obs) #converts from Lazy format to normal numpy array see wrappers_atari.py
timesteps=10000
for t in range(timesteps):
    if t%50==0:
        action = env.action_space.sample()
    else:
        Q = target_model.predict([obs[np.newaxis,:],info[np.newaxis,:]])[0]          # Q-values predictions

        action = np.argmax(Q)
    for i in range(4):
        next_obs, reward, done, info = env.step(action)     # result of action
        info = np.array(list(info.values()))
    next_obs = np.array(next_obs) #converts from Lazy format to normal numpy array see wrappers_atari.py
    obs=next_obs
    print(info)
    if done:
        obs = env.reset()           #restart game if done

