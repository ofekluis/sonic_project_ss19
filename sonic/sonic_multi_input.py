from keras import optimizers
import matplotlib.pyplot as plt
import scipy.stats as st
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
    training=int(sheet.cell(sheet.row_count,1).value)+1
    global training_folder
    training_folder='Training_'+str(training)
    retval = os.getcwd()
    print(retval)
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
    memory = deque(maxlen=40000)
    #global epsilon
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
    image_size = (128,128,frames_stack)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    converged=False # flag to check convergence (diff between Q and Q_target is small enough)
    total_mean=0
    total_var=0
    means=[]
    stds=[]
    rewards=[]
    plot_interval=20 # how often (in experiments) to plot graphs
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
        max_x = defaultdict(lambda: 0.0)
        avg_reward_List=[]
        total_total_rew=0
        global rewardList
        rewardList=[]
        steps=0
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
            env = retro.make(game, state, record="logs/"+training_folder)
            env = wrappers.WarpFrame(env, 128, 128, grayscale=True)
            env = wrappers.FrameStack(env,frames_stack)
            env = wrappers.SonicDiscretizer(env) # Discretize the environment for q learning
            env = wrappers.RewardWrapper(env) # custom reward calculation
            obs = np.array(env.reset()) #game start
            done = False
            total_raw_reward = 0.0
            Q= np.empty([])
            next_info=dict([])
            experimentRewardList=np.zeros(timesteps)
            train_interval=100
            gameList=[]
            stateList=[]
            minRewList=[]
            maxRewList=[]
            total_rewList=[]
            completed_levelList=[]
            current_max_x=0.0
            done = False
            lvl=None
            #Observation
            #in this loop sonic only plays according to epsilon greedy and saves its experience
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
                next_obs, reward, done, next_info = env.step(action)     # result of action

                next_info_dic=next_info
                next_info = np.array(list(next_info_dic.values()))
                total_raw_reward += reward
                max_reward = max(reward, max_reward)
                min_reward = min(reward, min_reward)
                lvl= (next_info_dic["zone"],next_info_dic["act"])
                if env.max_x > max_x[lvl]:
                    # if this is the farthest we've ever reached in this level
                    max_x[lvl] = env.max_x
                if steps>0:
                    memory.append((obs, next_obs ,action, reward, done, info, next_info))
                obs = next_obs
                info= next_info
                info_dic= next_info_dic
                current_max_x = max(current_max_x, env.max_x)

                #todo initalize vars
                #total_total_rew+=total_total_rew
                #total_steps=(e+1)*timesteps
                #total_avg_reward= total_total_rew/total_steps
                #avg_reward_List.append(total_avg_reward)
                experimentRewardList[i]=reward
                rewards.append(reward)
                #to calculate coinfidence interval
                #liste rewards von experiment
                #liste von allen experiementen (liste von zeile drüber(Listen))
                if steps>=mb_size:
                    if steps%train_interval==0:
                        #train on a random sample from memory of size mb_size
                        minibatch=random.sample(memory,mb_size)
                        info_inputs = np.zeros((mb_size,11))
                        inputs_shape = (mb_size,) + image_size
                        inputs = np.zeros(inputs_shape)
                        targets = np.zeros((mb_size, ACTION_SIZE))
                        for i,(obs,next_obs,action,reward,done,info,next_info) in enumerate(minibatch):
                            #reward = min(reward,reward_clip)
                            obs=np.array(obs)
                            next_obs=np.array(next_obs)
                            inputs[i] = obs
                            info_inputs[i] = info
                            # double Q
                            Q = model.predict([obs[np.newaxis,:],info[np.newaxis,:]])[0]          # Q-values predictions
                            Q_next = model.predict([next_obs[np.newaxis,:],next_info[np.newaxis,:]])[0]
                            Q_target_next = target_model.predict([next_obs[np.newaxis,:]
                                ,next_info[np.newaxis,:]])[0]
                            targets[i] = copy.copy(Q)
                            #diff+=sum(Q_next-Q_target_next)
                            if done:
                                targets[i, action] = reward
                            else:
                                targets[i, action] = reward + gamma * Q_target_next[np.argmax(Q_next)]
                        #train network on constructed inputs,targets
                        logs = model.train_on_batch([inputs, info_inputs], targets)
                        write_log(tensorboard, train_names, logs, steps)
                    if steps%target_step_interval==0:
                        #copy model weights to target model weights
                        target_model.set_weights(model.get_weights())
                    if steps%check_point_interval==0:
                        model.save_weights(retval+"/logs/"+training_folder+"/model_checkpoints/\
                                           sonic_model_"+str(steps)+".h5")
                if done:
                    obs=env.reset()
            #decay epsilon
            if epsilon > 0.005:
                epsilon*=epsilon_decay
            mean_e=np.mean(experimentRewardList)
            weight=1/(e+1)
            old_mean=total_mean
            total_mean=mean_e*weight+total_mean*(1-weight)
            var_e=np.var(experimentRewardList)
            total_var= var_e*weight+total_var*(1-weight)+weight*(1-weight)*(total_mean-old_mean)**2

            means.append(total_mean)
            stds.append(np.sqrt(total_var))
            if e>0 and e%plot_interval==0:
                #plot avg reward graph
                plot_avg_reward(means,stds,e,epsilon,timesteps,retval)
            print("Total reward: {}".format(total_raw_reward))
            print("Avg. step reward: {}".format(total_raw_reward/steps))
            print("Max distance on x axis: ", max_x[lvl])
            print("Current max distance on x axis: ", current_max_x)
            print("Experiment",e,"out of",experiments,"with",timesteps,"steps is finished")
            gameList.append(game)
            stateList.append(state)
            minRewList.append(min_reward)
            maxRewList.append(max_reward)
            total_rewList.append(total_raw_reward)
            completed_level=False
            if info_dic["level_end_bonus"] > 0:
                completed_level=True
            completed_levelList.append(completed_level)
            env.close()
    insertToSpreadSheets(training,gameList,stateList,eps,experiments,min_rewardList,maxRewList,total_rewList,timesteps,frames_stack,learning_rate,completed_levelList,mb_size)

def plot_avg_reward(means,stds, e,epsilon,timesteps,retval):
    ci = 0.95 # 95% confidence interval
    means=np.array(means)
    stds=np.array(stds)
    n = means.size

    # compute upper/lower confidence bounds
    crit_val = st.t.ppf((ci + 1) / 2, n)
    lb = means - crit_val * stds / np.sqrt(n)
    ub = means + crit_val * stds / np.sqrt(n)
    avg_reward=means[-1]
    print ('Avg. Reward per step after %d experiments: %.4f' % (e, avg_reward))

    # clear plot frame
    plt.clf()


    # plot average reward
    x = np.arange(0, (e+1)*timesteps, timesteps)
    plt.plot(x, means, color='blue', label="epsilon=%.2f" % epsilon)
    # plot upper/lower confidence bound
    plt.fill_between(x=x, y1=lb, y2=ub, color='blue', alpha=0.2, label="CI %.2f" % ci)

    plt.grid()
    plt.ylim(-2, 5) # limit y axis
    plt.title('Avg. Reward per step after %d experiments: %.4f' % (e, avg_reward))
    plt.ylabel("Reward per step")
    plt.xlabel("Steps")
    plt.savefig(retval+"/logs/"+training_folder+"/graphs/avg_reward_"+str(e)+".png",bbox_inches='tight')

def insertToSpreadSheets(training,gameList,stateList,eps,experiments,min_rewardList,maxRewList,total_rewList,timesteps,frames_stack,learning_rate,completed_levelList,mb_size):
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("Creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("SonicTable").sheet1  # Open the spreadhseet

    for e in experiments:
        insertRow=[training,gameList[e],stateList[e], eps, experiments,timesteps,min_rewardList[e],maxRewList[e],total_rewList[e],frames_stack,mb_size,learning_rate,completed_levelList[e]]
        sheet.append_row(insertRow)

    #flag= False
    #if flag ==False:
     #   print(sheet.cell(sheet.row_count,1).value)
      #  print(type(training))
       # flag ==True
        #insertRow = [training,game,state, eps,experiments,gamma,min_reward,max_reward,total_raw_reward,timesteps,learning_rate, frames_stack,completed_level]
        #sheet.resize(1)
        #sheet.append_row(insertRow)
    #else:
     #   insertRow = [training,game,state,eps,experiments,gamma,min_reward,max_reward,timesteps, learning_rate, frames_stack,completed_level]
        #sheet.resize(1)
      #  sheet.append_row(insertRow)


def convertBK2toMovie():
    os.chdir("logs/"+training_folder)
    directory = os.fsencode(os.getcwd())
    for file in os.listdir(directory):
        os.system('python3 convertbk2-mp4.py '+str(file))

if __name__ == '__main__':
    main()
    #convertBK2toMovie() #not working yet, needs to be done

    evaluationScript.main(rewardList,experiments,timesteps,eps,epsilon_decay)


