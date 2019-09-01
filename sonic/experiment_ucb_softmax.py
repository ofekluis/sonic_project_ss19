import gym_bandits
import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math

def epsilon_greedy(epsilon):
    rand = np.random.random()
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action

# number of rounds (iterations)
num_plays = 2000
epsilon = 0.05
num_experiments = 2001
arms = 10
c=0.5 # degree of exploration for ucb

## interactive plotting on (no need to close window for next iteration)
plt.ion()
plt.figure(figsize=(20,10))

#######################
# conduct experiment
#######################
# memorize all rewards of all episodes in rewardMemory for statistic plotting
rewardMemory = []
decay = 1e-10

for e in range (1, num_experiments, 1):

    # create new instance of bandit environment
    env = gym.make("BanditTenArmedGaussian-v0")
    env.reset()

    # count of number of times an arm was pulled
    arm_count = np.zeros(arms)
    # Q value => expected average reward
    Q = np.zeros(arms)
    rewards = np.zeros(num_plays)
    for i in range (1, num_plays, 1):
        max_ucb=0
        # choose arm according to max ucb
        for j in range(0, arms):
            if arm_count[j] > 0:
                ucb = Q[j] + c*math.sqrt(math.log(i) / arm_count[j])
            else:
                ucb = math.inf # if never visited, ucb should be big
            if ucb>max_ucb:
                max_ucb= ucb
                arm = j
        #arm = epsilon_greedy(epsilon)
        # get reward/observation/terminalInfo
        observation, reward, done, info = env.step(arm)
        # update the count of that arm
        arm_count[arm] += 1
        # recalculate its Q value
        Q[arm] = Q[arm]+ (1/arm_count[arm])*(reward-Q[arm])
        # memorize rewards per play
        c = c-decay if c > 0 else 0
        rewards[i] = reward

    # memorize reward array
    rewardMemory.append(rewards)

    ##################################################
    # Live plotting of statistics every 100 episodes
    ##################################################
    if e%200 == 0:

        ci = 0.95 # 95% confidence interval
        means = np.mean(rewardMemory, axis=0)
        stds = np.std(rewardMemory, axis=0)
        n = means.size

        # compute upper/lower confidence bounds
        test_stat = st.t.ppf((ci + 1) / 2, e)
        lower_bound = means - test_stat * stds / np.sqrt(e)
        upper_bound = means + test_stat * stds / np.sqrt(e)

        print ('Avg. Reward per step in experiment %d: %.4f' % (e, sum(means) / num_plays))

        # clear plot frame
        plt.clf()

        # plot average reward
        plt.plot(means, color='blue', label="epsilon=%.2f" % epsilon)

        # plot upper/lower confidence bound
        x = np.arange(0, num_plays, 1)
        plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)

        plt.grid()
        plt.ylim(0, 2) # limit y axis
        plt.title('Avg. Reward per step (UCB with c=%.2f, decay=%.1e) in experiment %d: %.4f' % (c ,decay, e, sum(means) / num_plays))
        plt.ylabel("Reward per step")
        plt.xlabel("Play")
        plt.legend()
        plt.show()
        plt.pause(0.01)

## disable interactive plotting => otherwise window terminates
plt.ioff()
plt.show()
