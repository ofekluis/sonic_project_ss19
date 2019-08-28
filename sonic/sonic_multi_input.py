import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from collections import deque
from keras.callbacks import TensorBoard
from skimage import color
from skimage.transform import resize
#import gym_remote.exceptions as gre
import os
import random
import numpy as np
import atari_wrappers
import gym
import retro
import copy
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
#from retro_contest.local import make
import effnet
import time
import tensorflow as tf
from datetime import timedelta
from sklearn.metrics import mean_squared_error
ACTION_SIZE=8
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

#env specific
class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

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

#end of env specific
def reward_calc(last_info, info, done):
    """calculates the reward from the info vector of the two last timesteps"""
    if len(last_info)==0: #in case this is the first step, we have no reward
        return 0
    rew=0
    if done:
        return 1000
    if info["x"] > last_info["x"]:
        rew+=10
    if info["lives"] < last_info["lives"]:
        rew-=100
    if info["rings"] > last_info["rings"]:
        rew+=10
    return rew

def main():

    start_time = time.time()

    games = ["SonicTheHedgehog-Genesis"]
    #game = np.random.choice(games,1)[0]
    #env = retro.make(game, state,record='logs/')
    #env = AllowBacktracking(make(game, state)) #contest version

    #writing to spreadsheets
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("Creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("SonicTable").sheet1  # Open the spreadhseet
    data = sheet.get_all_records()  # Get a list of all records
    numRows = sheet.row_count+1  # Get the number of rows in the sheet
    print(numRows)
    row = sheet.row_values(2)
    print(row)
    insertRow = [1, 7]
    sheet.resize(1)
    sheet.append_row(insertRow)
    pprint (data)
    #writing to spreadsheets

    # Parameters
    timesteps = 10000#4500
    memory = deque(maxlen=30000)
    epsilon = 0.5                                #probability of doing a random move
    max_random = 1
    min_random = 0.1                           #minimun randomness #r12
    rand_decay = 1e-3                                #reduce the randomness by decay/loops
    gamma = 0.99                               #discount for future reward
    mb_size = 256                               #learning minibatch size
    loops = 10#45                               #loop through the different game levels
    sub_loops = 10#100
    hold_action = 1                            #nb frames during which we hold (4 for normal, 1 for contest)
    learning_rate = 5e-5
    max_reward = 0
    min_reward = 10000
    frames_stack=4 # how many frames to be stacked together
    #action_threshold = 1
    target_step_interval = 10
    reward_clip = 200
    resize_to = [128,128]
    save_factor= 500 # when to save the model according to loops_count
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        model=effnet.Effnet(input_shape=(84,84,frames_stack),nb_classes=ACTION_SIZE, info=11)
        if os.path.isfile("sonic_model.h5"):
            model.load_weights("sonic_model.h5")

        model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

        tensorboard = TensorBoard(log_dir="logs/sonic_modmemdecayrdq18_reshape_64x512mb256_resc_target_interval_{}_memory_30000_lr_{}_decay_{}.{}".format(target_step_interval,learning_rate, rand_decay, time.time()))
        tensorboard.set_model(model)
        train_names = ["Loss", "Accuracy"]

        # serialize model to JSON
        model_json = model.to_json()
        with open("sonic_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("sonic_model.h5")
        model.save_weights("sonic_target_model.h5")

        #env.close()
    for training_loop in range(loops):
        with tf.Session(config=config) as sess:
            model = model_from_json(model_json)
            model.load_weights("sonic_model.h5")
            model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
            target_model = model_from_json(model_json)
            target_model.load_weights("sonic_target_model.h5")
            target_model.trainable = False
            target_model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

            for sub_training_loop in range(sub_loops):
                loop_start_time = time.time()
                game = np.random.choice(games,1)[0]
                state = np.random.choice(retro.data.list_states(game),1)[0]
                while state=="GreenHillZone.Act1":
                    state = np.random.choice(retro.data.list_states(game),1)[0]
                print("Playing",game,"-",state)
                #env = AllowBacktracking(retro.make(game, state,scenario="scenario.json", record="logs/"))
                env = retro.make(game, state,scenario="scenario.json", record="logs/")
                env = atari_wrappers.WarpFrame(env, 84, 84, grayscale=True)
                env = atari_wrappers.FrameStack(env,frames_stack)
                env = SonicDiscretizer(env) #contest version
                obs = env.reset() #game start
                done = False
                total_raw_reward = 0.0
                Q= np.empty([])
                last_info=dict([])
                #Observation
                for t in range(timesteps):
                    #env.render() #display training
                    if np.random.rand() > epsilon and Q.size==ACTION_SIZE:
                        #pick a random action
                        action = np.argmax(Q)
                    else:
                        action = env.action_space.sample()
                    reward_hold = np.zeros(hold_action)
                    for h in range(hold_action):
                        obs, reward_hold[h], done, info = env.step(action)     # result of action
                    reward = reward_calc(last_info,info,done)
                    info_dic=info
                    obs = np.array(obs) #converts from Lazy format to normal numpy array see wrappers_atari.py
                    #reward = sum(reward_hold)
                    reward_ = min(reward,reward_clip)
                    info = np.array(list(info.values()))
                    #Bellman double Q
                    Q = model.predict([obs[np.newaxis,:],info[np.newaxis,:]])          # Q-values predictions
                    Q_target = target_model.predict([obs[np.newaxis,:],info[np.newaxis,:]])

                    target_ = copy.copy(Q)

                    if done:
                        target_[0,action] = reward_ - reward_clip
                    else:
                        target_[0,action] = reward_ + gamma * Q_target[0,:][np.argmax(Q[0,:])]

                    distance_from_target = mean_squared_error(Q, target_)
                    total_raw_reward += reward

                    max_reward = max(reward, max_reward)
                    min_reward = min(reward, min_reward)

                    if distance_from_target > 25:
                        memory.append((obs, action, reward, done, info))
                    elif done:
                        memory.append((obs, action, reward, done, info))


                    if done:
                        obs = env.reset()           #restart game if done
                    last_info=info_dic
                #epsilon = min_random + (max_random-min_random)*np.exp(-rand_decay*(training_loop*sub_loops + sub_training_loop+1))
                epsilon -=0.001
                print("Total reward: {}".format(total_raw_reward))
                print("Avg. step reward: {}".format(total_raw_reward/timesteps))
                print("Observation Finished",sub_training_loop+1,"x",training_loop+1,"out of",sub_loops,"x",loops)

                # Learning
                if len(memory) >= mb_size:
                    minibatch_train_start_time = time.time()

                    #sample memory
                    minibatch = random.sample(memory, mb_size)

                    info_inputs = np.zeros((mb_size,11))
                    inputs_shape = (mb_size,) + obs.shape
                    inputs = np.zeros(inputs_shape)
                    targets = np.zeros((mb_size, env.action_space.n))

                    #double Q: fix the target for a time to stabilise the model
                    if (sub_training_loop+1)%target_step_interval == 0:# and training_loop*sub_loops + sub_training_loop+1 >= 100: #r12 chase score first
                        #with tf.device('/cpu:0'):
                        # serialize weights to HDF5
                        #model.save_weights("sonic_model.h5")
                        model.save_weights("sonic_target_model.h5")
                        loops_count= training_loop*sub_loops + sub_training_loop+1
                        if loops_count % save_factor == 0:
                            model.save_weights("sonic_model_"+ loops__count+".h5")
                        # create model from json and do inference on cpu
                        target_model = model_from_json(model_json)
                        target_model.load_weights("sonic_target_model.h5")
                        target_model.trainable = False
                        target_model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

                    for i in range(0, mb_size):
                        obs = minibatch[i][0]
                        action = minibatch[i][1]
                        reward = minibatch[i][2]
                        done = minibatch[i][3]
                        info = minibatch[i][4]

                        #reward clipping
                        reward = min(reward,reward_clip)

                        #Bellman double Q
                        inputs[i] = obs[np.newaxis,:]
                        info_inputs[i] = info
                        Q = model.predict([obs[np.newaxis,:],info[np.newaxis,:]])          # Q-values predictions
                        Q_target = target_model.predict([obs[np.newaxis,:],info[np.newaxis,:]])

                        targets[i] = copy.copy(Q)

                        if done:
                            targets[i, action] = reward - reward_clip
                        else:
                            targets[i, action] = reward + gamma * Q_target[0,:][np.argmax(Q[0,:])]

                    #train network on constructed inputs,targets
                    logs = model.train_on_batch([inputs, info_inputs], targets)
                    write_log(tensorboard, train_names, logs, training_loop*sub_loops + sub_training_loop)

                    model.save_weights("sonic_model.h5")

                    print("Model minibatch training lasted:",
                          str(timedelta(seconds=time.time()-minibatch_train_start_time)),"dd:hh:mm:ss")
                    print("Learning Finished",sub_training_loop+1,"x",training_loop+1,"out of",sub_loops,"x",loops)

                env.close()

                print("Loop lasted:",str(timedelta(seconds=time.time()-loop_start_time)),"dd:hh:mm:ss")
                print("Training lasted:",str(timedelta(seconds=time.time()-start_time)),"dd:hh:mm:ss")
                print("Rewards between",min_reward,"and",max_reward)
                print("Percentage of random movements set to", epsilon * 100, "%\n")

if __name__ == '__main__':
    #try:
        main()
    #except gre.GymRemoteError as exc:
    #    print('exception', exc)


