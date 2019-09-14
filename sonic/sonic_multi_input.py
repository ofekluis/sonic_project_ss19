from keras import optimizers
import matplotlib.pyplot as plt
import scipy.stats as st
from keras.models import load_model
from keras.models import model_from_json
from collections import deque,defaultdict
from keras.callbacks import TensorBoard
from skimage import color
from skimage.transform import resize
import evaluationScript
import os
import random
import numpy as np
import wrappers
import gym
import retro
import time
import copy
import model as m
import time
import tensorflow as tf
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

ACTION_SIZE=8 # number of different actions we use for sonic
def write_log(callback, names, logs, batch_no):
    # tensorboard log stuff
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def main(epsilon,experiments,timesteps,mb_size,frames_stack):

    start_time = time.time()

    games = ["SonicTheHedgehog-Genesis"]

    #delete old weights
    #os.remove("sonic_model_0.h5")
    #os.remove("sonic_model.h5")
    #os.remove("sonic_target_model.h5")

    #writing to spreadsheets
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("Creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("SonicTable").sheet1  # Open the spreadhseet
    data = sheet.get_all_records()  # Get a list of all records
    numRows = sheet.row_count  # Get the number of rows in the sheet
    row = sheet.row_values(2)
    sheet.resize(numRows)
    training=int(sheet.cell(sheet.row_count,1).value)+1 # get the training number for listing in google sheets
    insertRow=[training]
    global training_folder
    training_folder='Training_'+str(training)
    retval = os.getcwd()
    sheet.append_row(insertRow)
    #make some dirs for logs
    os.chdir(retval+"/logs")
    if not os.path.isdir(training_folder):
        os.mkdir(training_folder)
    os.chdir(retval+"/logs/"+training_folder)
    if not os.path.isdir("model_checkpoints"):
        os.mkdir("model_checkpoints")
    if not os.path.isdir("graphs"):
        os.mkdir("graphs")


    os.chdir("../..")

    # Parameters
    #global timesteps
    #timesteps = 1000#4500
    memory = deque(maxlen=40000) # memory for saving observations, as a queue, maxlen kept low for ram reasons
    #global epsilon
    action_persist=4 # how many environments steps to persist for an action
    eps = epsilon
    global epsilon_decay                               #probability of doing a random move
    epsilon_decay = 0.999  #will be multiplied with epsilon for decaying it
    gamma = 0.99                               #discount for future reward
    #mb_size = 256
    #global experiments                             #learning minibatch size
    #experiments = 3 #number of experiments to run
    learning_rate = 5e-5
    max_reward = 0
    min_reward = 10000
    check_point_interval=100000
    #frames_stack=4 # how many frames to be stacked together
    #action_threshold = 1
    target_step_interval = 8192
    reward_clip = 1000 #maximum reward allowed for step
    image_size = (128,128,frames_stack) # image size
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    converged=False # flag to check convergence (diff between Q and Q_target is small enough)
    total_mean=0
    total_var=0
    means=[] # Rewards mean list after each experimet
    stds=[] # Rewards standard deviation list after each experiment
    plot_interval=20 # how often (in experiments) to plot graphs
    max_x = defaultdict(lambda: 0.0) #keep track of maxium x distance covered in a level
    total_total_rew=0
    steps=0 # how many steps did sonic do in total over all experiments
    with tf.Session(config=config) as sess:
        model=m.ddqn_model(input_shape=(128,128,frames_stack),nb_classes=ACTION_SIZE, info=11)
        target_model=m.ddqn_model(input_shape=(128,128,frames_stack),nb_classes=ACTION_SIZE, info=11)
        if os.path.isfile("sonic_model.h5"):
            model.load_weights("sonic_model.h5")
        target_model.set_weights(model.get_weights())
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
        target_model.compile(loss="mse", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
        target_model.trainable = False
        tensorboard = TensorBoard(log_dir="logs/"+training_folder+"/sonic_target_interval_{}_memory_40000_lr_{}_epsilon_{}.{}".format(target_step_interval,learning_rate, epsilon, time.time()))
        tensorboard.set_model(model)
        train_names = ["Loss", "Accuracy"]

        game = "SonicTheHedgehog-Genesis"#np.random.choice(games,1)[0]
        #train on all but the first level, which is reserved for testing
        states = retro.data.list_states(game)[1:]
        #information for tracking below
        gameList=[]
        stateList=[]
        minRewList=[]
        maxRewList=[]
        total_rewList=[]
        completed_levelList=[]
        train_interval=100 # how often to train in steps
        for e in range(experiments):
            """
            if converged:
                # model converged
                model.save_weights("sonic_target_model.h5")
                print("Model converged, stopping training")
                break
            """
            loop_start_time = time.time()
            #pick a level to train on randomly
            state = np.random.choice(states,1)[0]
            print("Playing",game,"-",state)
            # apply all needed wrappers
            env = retro.make(game, state, record="logs/"+training_folder) #make environment
            env = wrappers.WarpFrame(env, 128, 128, grayscale=True) #scales frame
            env = wrappers.FrameStack(env,frames_stack) #collects 4 frames as one
            env = wrappers.SonicDiscretizer(env) # Discretize the environment for q learning
            env = wrappers.RewardWrapper(env) # custom reward calculation
            obs = np.array(env.reset()) #game start
            total_raw_reward = 0.0 # total reward for level in this experiment
            Q= np.empty([]) # initialize Q vector
            next_info=dict([]) # keep track of next info vector
            experimentRewardList=np.zeros(timesteps)
            current_max_x=0.0 # current level max distance in this experiment
            done = False # keep track if sonic lost all his lives
            lvl=None
            #Observation
            #in this loop sonic plays according to epsilon greedy and saves its experience
            #it also trains on random batch every train_interval steps
            for i in range(timesteps):
                steps=e*timesteps+i
                #env.render() #display training
                if np.random.rand() > epsilon and e>0:
                    Q = model.predict([np.array(obs)[np.newaxis,:],info[np.newaxis,:]])[0]          # Q-values predictions
                    action = np.argmax(Q)
                else:
                    #pick a random action
                    action = env.action_space.sample()
                rewards=np.zeros(action_persist)
                for j in range(action_persist):
                    won, next_obs, rewards[j], done, next_info = env.step(action)     # result of action
                    if done: break
                reward=sum(rewards) # sum the rewards of doing action, action_persist times
                if j>0: reward=reward/j #scale reward
                next_info_dic=next_info
                next_info = np.array(list(next_info_dic.values())) # make array out of lazy represenation see wrappers.py
                total_raw_reward += reward
                #keep track of min and max reward
                max_reward = max(reward, max_reward)
                min_reward = min(reward, min_reward)
                lvl= (next_info_dic["zone"],next_info_dic["act"])
                if env.max_x > max_x[lvl]:
                    # if this is the farthest we've ever reached in this level
                    # mostly to keep track
                    max_x[lvl] = env.max_x
                if steps>0:
                    # collect the observation into the memory queue
                    memory.append((obs, next_obs ,action, reward, done, info, next_info))
                # keep track of last observation and info vector for training
                obs = next_obs
                info= next_info
                info_dic= next_info_dic
                current_max_x = max(current_max_x, env.max_x) # get current maximum x distance
                experimentRewardList[i]=reward # save rewards of current experiment
                if steps>=mb_size:
                    # first make sure we have enough experience for a minibatch
                    # training
                    if steps%train_interval==0:
                        #train on a random batch sampled from memory of size mb_size
                        minibatch=random.sample(memory,mb_size)
                        # initialize input and target vectors for the training
                        info_inputs = np.zeros((mb_size,11))
                        inputs_shape = (mb_size,) + image_size
                        inputs = np.zeros(inputs_shape)
                        targets = np.zeros((mb_size, ACTION_SIZE))
                        for i,(obs,next_obs,action,reward,done,info,next_info) in enumerate(minibatch):
                            #reward = min(reward,reward_clip)
                            obs=np.array(obs) # convert lazy format to array, see wrappers.py
                            next_obs=np.array(next_obs)
                            inputs[i] = obs #collect inputs
                            info_inputs[i] = info
                            # predict Q values for the double Q update
                            Q = model.predict([obs[np.newaxis,:],info[np.newaxis,:]])[0]          # Q-values predictions
                            Q_next = model.predict([next_obs[np.newaxis,:],next_info[np.newaxis,:]])[0]
                            Q_target_next = target_model.predict([next_obs[np.newaxis,:]
                                ,next_info[np.newaxis,:]])[0]
                            targets[i] = copy.copy(Q) #target is former Q plus the update below
                            #diff+=sum(Q_next-Q_target_next)
                            if done:
                                # in case of a finished episode
                                targets[i, action] = reward
                            else:
                                # otherwise do double Q update
                                targets[i, action] = reward + gamma * Q_target_next[np.argmax(Q_next)]
                        #train network on constructed inputs,targets
                        logs = model.train_on_batch([inputs, info_inputs], targets)
                        write_log(tensorboard, train_names, logs, steps)
                    if steps%target_step_interval==0:
                        #copy model weights to target model weights
                        target_model.set_weights(model.get_weights())
                    if steps%check_point_interval==0:
                        #save weights every check_point_interval steps
                        model.save_weights(retval+"/logs/"+training_folder+"/model_checkpoints/sonic_model_"+str(steps)+".h5")
                if done:
                    #in case sonic looses/wins start again
                    obs=env.reset()
            #decay epsilon
            if epsilon > 0.005:
                epsilon*=epsilon_decay
            # additivly keep track of mean and standard error for plotting
            # worth mentioning that this would blow up if all rewards were kept
            # in memory
            mean_e=np.mean(experimentRewardList)
            weight=1/(e+1)
            old_mean=total_mean
            total_mean=mean_e*weight+total_mean*(1-weight)
            var_e=np.var(experimentRewardList)
            total_var= var_e*weight+total_var*(1-weight)+weight*(1-weight)*(total_mean-old_mean)**2

            #add new mean and se to their lists
            means.append(total_mean)
            stds.append(np.sqrt(total_var))
            if e>0 and e%plot_interval==0:
                #plot avg reward graph every plot_interval experiments
                plot_avg_reward(means,stds,e,epsilon,timesteps,retval)
            print("Total reward: {}".format(total_raw_reward))
            print("Avg. step reward: {}".format(total_raw_reward/steps))
            print("Max distance on x axis: ", max_x[lvl])
            print("Current max distance on x axis: ", current_max_x)
            print("Current epsilon:",epsilon)
            print("Experiment",e,"out of",experiments,"with",timesteps,"steps is finished")
            # some more info for tracking
            gameList.append(game)
            stateList.append(state)
            minRewList.append(min_reward)
            maxRewList.append(max_reward)
            total_rewList.append(total_raw_reward)
            completed_level=False
            #if info_dic["level_end_bonus"] > 0:
             #   completed_level=True
            completed_levelList.append(won)
            env.close()
    retval = os.getcwd()
    os.chdir(retval+"/logs/"+training_folder)
    if won:
        retval=os.getcwd()
        files = os.listdir(retval)
        paths = [os.path.join(retval, basename) for basename in files if basename.endswith('000000.bk2')]
        latest_file=max(paths, key=os.path.getctime)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        os.rename(os.path.basename(latest_file), os.path.basename(latest_file)+"_LEVEL_COMPLETED_"+timestr)
    os.chdir("../..")
    insertToSpreadSheets(training,gameList,stateList,eps,experiments,minRewList,maxRewList,total_rewList,timesteps,frames_stack,learning_rate,completed_levelList,mb_size)

def plot_avg_reward(means,stds, e,epsilon,timesteps,retval):
    #plots the avg. reward graph
    ci = 0.95 # 95% confidence interval
    means=np.array(means)
    stds=np.array(stds)
    n = means.size

    # compute upper/lower confidence bounds
    crit_val = st.t.ppf((ci + 1) / 2, n)
    lb = means - crit_val * stds / np.sqrt(n)
    ub = means + crit_val * stds / np.sqrt(n)
    #get current mean
    avg_reward=means[-1]
    print ('Avg. reward per step after %d experiments: %.4f' % (e+1, avg_reward))

    # clear plot frame
    plt.clf()


    # arange x axis according to the number of steps in an experiment
    x = np.arange(0, (e+1)*timesteps, timesteps)
    # plot average reward
    plt.plot(x, means, color='blue', label="epsilon=%.2f" % epsilon)
    # plot upper/lower confidence bound
    plt.fill_between(x=x, y1=lb, y2=ub, color='blue', alpha=0.2, label="CI %.2f" % ci)

    #build/draw graph
    plt.grid()
    plt.ylim(-2, 5) # limit y axis
    plt.title('Avg. reward per step after %d experiments: %.4f' % (e+1, avg_reward))
    plt.ylabel("Avg. reward per step")
    plt.xlabel("Steps")
    #save graph to the logs dir
    plt.savefig(retval+"/logs/"+training_folder+"/graphs/avg_reward_"+str(e)+".png",bbox_inches='tight')

def insertToSpreadSheets(training,gameList,stateList,eps,experiments,min_rewardList,maxRewList,total_rewList,timesteps,frames_stack,learning_rate,completed_levelList,mb_size):
    # insert tracking information to our online google sheets table.
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("Creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("SonicTable").sheet1  # Open the spreadhseet

    #training2=int(sheet.cell(sheet.row_count,1).value)+1
    for e in experiments:
        insertRow=[training,gameList[e],stateList[e], eps, experiments,timesteps,min_rewardList[e],maxRewList[e],total_rewList[e],frames_stack,mb_size,learning_rate,completed_levelList[e]]
        sheet.append_row(insertRow)


def convertBK2toMovie():
    # converts logs files into sonic videos
    os.chdir("logs/"+training_folder)
    directory = os.fsencode(os.getcwd())
    for file in os.listdir(directory):
        os.system('python3 convertbk2-mp4.py '+str(file))

if __name__ == '__main__':
    main()
    #convertBK2toMovie() #not working yet, needs to be done
    #evaluationScript.main(rewardList,experiments,timesteps,eps,epsilon_decay)


