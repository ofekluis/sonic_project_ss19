from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
from collections import deque,defaultdict
from keras.callbacks import TensorBoard
from skimage import color
from skimage.transform import resize
#import gym_remote.exceptions as gre
import evaluationScript
import os
import random
import numpy as np
import wrappers
import gym
import retro
import copy
#from retro_contest.local import make
import model as m
import time
import tensorflow as tf
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

ACTION_SIZE=8
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


#end of env specific
def main():

    start_time = time.time()

    games = ["SonicTheHedgehog-Genesis"]

    #writing to spreadsheets
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("Creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("SonicTable").sheet1  # Open the spreadhseet
    data = sheet.get_all_records()  # Get a list of all records
    numRows = sheet.row_count  # Get the number of rows in the sheet
    print(numRows)
    row = sheet.row_values(2)
    #insertRow = [1, 7]
    sheet.resize(numRows)
    #sheet.append_row(insertRow)
    #pprint (data)

    # Parameters
    timesteps = 10000#4500
    memory = deque(maxlen=30000)
    epsilon = 0.3                                #probability of doing a random move
    epsilon_decay = 0.999  #will be multiplied with epsilon for decaying it
    max_random = 1
    min_random = 0.1                           #minimun randomness #r12
    rand_decay = 1e-3                                #reduce the randomness by decay/loops
    gamma = 0.99                               #discount for future reward
    mb_size = 256                               #learning minibatch size
    loops = 35#45                               #loop through the different game levels
    sub_loops = 35#100
    learning_rate = 5e-5
    max_reward = 0
    min_reward = 10000
    frames_stack=4 # how many frames to be stacked together
    #action_threshold = 1
    target_step_interval = 10
    reward_clip = 1000 #maximum reward allowed for step
    image_size = (128,128,frames_stack)
    save_factor= 500000 # when to save the model according in game steps
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    converged=False # flag to check convergence (diff between Q and Q_target is small enough)
    with tf.Session(config=config) as sess:
        model=m.ddqn_model(input_shape=(128,128,frames_stack),nb_classes=ACTION_SIZE, info=11)
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
    game = "SonicTheHedgehog-Genesis"#np.random.choice(games,1)[0]
    #train on all but the first level, which is reserved for testing
    states = retro.data.list_states(game)[1:]
    max_x = defaultdict(lambda: 0.0)
    for training_loop in range(loops):
        if converged:
            # model converged
            model.save_weights("sonic_target_model.h5")
            print("Model converged, stopping training")
            break
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
                #pick a level to train on randomly
                state = np.random.choice(states,1)[0]
                print("Playing",game,"-",state)
                env = retro.make(game, state,scenario="scenario.json", record="logs/")
                env = wrappers.WarpFrame(env, 128, 128, grayscale=True)
                env = wrappers.FrameStack(env,frames_stack)
                env = wrappers.SonicDiscretizer(env) # Discretize the environment for q learning
                env = wrappers.RewardWrapper(env) # custom reward calculation
                obs = np.array(env.reset()) #game start
                done = False
                total_raw_reward = 0.0
                Q= np.empty([])
                last_info=dict([])
                #Observation
                #in this loop sonic only plays according to epsilon greedy and saves its experience
                for t in range(timesteps):
                    #env.render() #display training
                    if np.random.rand() > epsilon and sub_training_loop>0:
                        Q = model.predict([np.array(obs)[np.newaxis,:],info[np.newaxis,:]])[0]          # Q-values predictions
                        action = np.argmax(Q)
                    else:
                        #pick a random action
                        action = env.action_space.sample()
                    next_obs, reward, done, info = env.step(action)     # result of action
                    info_dic=info
                    info = np.array(list(info_dic.values()))
                    reward_ = min(reward,reward_clip)

                    total_raw_reward += reward
                    max_reward = max(reward, max_reward)
                    min_reward = min(reward, min_reward)
                    lvl= (info_dic["zone"],info_dic["act"])
                    if env.max_x > max_x[lvl]:
                        # if this is the farthest we've ever reached in this level
                        max_x[lvl] = env.max_x
                        reward+=20
                    memory.append((obs, next_obs ,action, reward, done, info))

                    obs = next_obs
                    if done:
                        obs = env.reset()           #restart game if done
                    last_info=info_dic
                #decay epsilon
                if epsilon > 0.005:
                    epsilon*=epsilon_decay
                print("Total reward: {}".format(total_raw_reward))
                print("Avg. step reward: {}".format(total_raw_reward/timesteps))
                print("Max distance on x axis: ", max_x[(info_dic["zone"],info_dic["act"])])
                print("Current max distance on x axis: ", env.max_x)
                print("Observation Finished",sub_training_loop+1,"x",training_loop+1,"out of",sub_loops,"x",loops)

                # Learning
                if len(memory) >= mb_size:
                    minibatch_train_start_time = time.time()

                    #sample memory
                    minibatch = random.sample(memory, mb_size)

                    info_inputs = np.zeros((mb_size,11))
                    inputs_shape = (mb_size,) + image_size
                    inputs = np.zeros(inputs_shape)
                    targets = np.zeros((mb_size, env.action_space.n))

                    # load a new target model every target_step_interval
                    # training loops and leave it unchanged 'till the lext
                    # interval
                    if (sub_training_loop+1)%target_step_interval == 0:# and training_loop*sub_loops + sub_training_loop+1 >= 100: #r12 chase score first
                        model.save_weights("sonic_target_model.h5")
                        loops_count= training_loop*sub_training_loop*timesteps
                        if loops_count % save_factor == 0:
                            model.save_weights("sonic_model_"+ str(loops_count)+".h5")
                        target_model = model_from_json(model_json)
                        target_model.load_weights("sonic_target_model.h5")
                        target_model.trainable = False
                        target_model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
                    diff=0
                    #preparing batch to send into the NN for learning, using bellman's double Q
                    #for the Q estimation
                    for i,(obs,next_obs,action,reward,done,info) in enumerate(minibatch):
                        reward = min(reward,reward_clip)
                        obs=np.array(obs)
                        next_obs=np.array(next_obs)
                        inputs[i] = obs
                        info_inputs[i] = info
                        # double Q
                        Q = model.predict([obs[np.newaxis,:],info[np.newaxis,:]])[0]          # Q-values predictions
                        Q_next = model.predict([next_obs[np.newaxis,:],info[np.newaxis,:]])[0]
                        Q_target = target_model.predict([next_obs[np.newaxis,:],info[np.newaxis,:]])[0]
                        targets[i] = copy.copy(Q)
                        diff+=sum(Q-Q_target)
                        if done:
                            targets[i, action] = reward - reward_clip
                        else:
                            targets[i, action] = reward + gamma * Q_target[np.argmax(Q_next)]
                    #train network on constructed inputs,targets
                    logs = model.train_on_batch([inputs, info_inputs], targets)
                    write_log(tensorboard, train_names, logs, training_loop*sub_loops + sub_training_loop)

                    model.save_weights("sonic_model.h5")

                    print("Model minibatch training lasted:",
                          str(timedelta(seconds=time.time()-minibatch_train_start_time)),"dd:hh:mm:ss")
                    print("Learning Finished",sub_training_loop+1,"x",training_loop+1,"out of",sub_loops,"x",loops)

                    if diff/mb_size < 0.000000000000001 and training_loop > 5:
                        # if there is not much difference in a batch after some
                        # training assume convergance
                        converged = True

                env.close()

                print("Loop lasted:",str(timedelta(seconds=time.time()-loop_start_time)),"dd:hh:mm:ss")
                print("Training lasted:",str(timedelta(seconds=time.time()-start_time)),"dd:hh:mm:ss")
                print("Rewards between",min_reward,"and",max_reward)
                print("Percentage of random movements set to", epsilon * 100, "%\n")
                flag= False
                if flag ==False:
                    print(sheet.cell(sheet.row_count,1).value)
                    training=int(sheet.cell(sheet.row_count,1).value)+1
                    print(type(training))
                    flag ==True
                    insertRow = [training,game,state, epsilon,loops,sub_loops,gamma,min_reward,max_reward,timesteps,learning_rate, frames_stack]
                    #sheet.resize(1)
                    sheet.append_row(insertRow)
                else:
                    insertRow = [training,game,state,epsilon,loops,sub_loops,gamma,min_reward,max_reward,timesteps, learning_rate, frames_stack]
                    #sheet.resize(1)
                    sheet.append_row(insertRow)


if __name__ == '__main__':
    main()
    evaluationScript.main()


