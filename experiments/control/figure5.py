from utils.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from RlGlue import RlGlue
from agents.DQNAgent import DQN
from environments.MountainCar import MountainCar

from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper
from utils.RatioMap import RatioMap

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

RUNS = 10
EPISODES = 500
RATIO_STEP = 10
MIN_RATIO = 10
MAX_RATIO = 80
SAMPLE_SIZE = 32

LEARNERS = [DQN, ]
action_dict = []

COLORS = {
    'QLearning': 'blue',
    'QRC': 'purple',
    'QC': 'green',
    'DQN': 'red',
}

# use stepsizes found in parameter study
STEPSIZES = {
    'QLearning': 0.003906,
    'QRC': 0.0009765,
    'QC': 0.0009765,
    'DQN': 0.0009765,
}

collectorb = Collector()
collectors = Collector()
collectorf = Collector()
collectorreward = Collector()
collectorbq = Collector()
collectorsq = Collector()
collectorfq = Collector()


def run_for_ratio(ratio_map, identifier):
    for run in range(RUNS):
        for Learner in LEARNERS:
            np.random.seed(run)
            torch.manual_seed(run)

            env = MountainCar()

            learner = Learner(env.features, env.num_actions, {
                'alpha': STEPSIZES[Learner.__name__],
                'epsilon': 0.1,
                'beta': 1.0,
                'target_refresh': 1,
                'buffer_size': 3000,
                'h1': 32,
                'h2': 32,
                'ratioMap': ratio_map,
                'sampleSize': SAMPLE_SIZE
            })

            agent = RlGlueCompatWrapper(learner, gamma=0.99)

            # print(agent.agent.target_net.fc_out.weight)
            glue = RlGlue(agent, env)

            glue.start()
            for episode in range(EPISODES):
                glue.num_steps = 0
                glue.total_reward = 0
                glue.runEpisode(max_steps=5000)

                print(Learner.__name__ + identifier, run, episode, glue.num_steps)

                collectorb.collect(
                    Learner.__name__ + identifier, agent.action_dict['back'] / 100)
                collectors.collect(
                    Learner.__name__ + identifier, agent.action_dict['stay'] / 100)
                collectorf.collect(
                    Learner.__name__ + identifier, agent.action_dict['forward'] / 100)
                collectorreward.collect(
                    Learner.__name__ + identifier, glue.total_reward)
                collectorbq.collect(
                    Learner.__name__ + identifier, sum(agent.agent.back_values) / len(agent.agent.back_values))
                collectorsq.collect(
                    Learner.__name__ + identifier, sum(agent.agent.stay_values) / len(agent.agent.stay_values))
                collectorfq.collect(
                    Learner.__name__ + identifier, sum(agent.agent.forward_values) / len(agent.agent.forward_values))

            action_dict.append(agent.action_dict)
            collectorb.reset()
            collectors.reset()
            collectorf.reset()
            collectorreward.reset()
            collectorbq.reset()
            collectorsq.reset()
            collectorfq.reset()
            agent.resetDict()


start_time = time.time()

ratioList = []
identifierList = []
for weight in range(MIN_RATIO, MAX_RATIO + 1, RATIO_STEP):
    ratioList += [RatioMap((100 - weight) / 2, weight, (100 - weight) / 2)]
    identifierList += ["-Stay-" + str(weight)]

for i in range(len(ratioList)):
    run_for_ratio(ratioList[i], identifierList[i])

''' plot '''
# Plot # actions
plt.figure()
ax = plt.gca()
for Learner in LEARNERS:
    for identifier in identifierList:
        name = Learner.__name__ + identifier
        # datab = collectorb.getStats(name)
        datas = collectors.getStats(name)
        # dataf = collectorf.getStats(name)
        # plot(ax, datab, label="back", color="red")
        plot(ax, datas, label=name + "do nothing")
        # plot(ax, dataf, label="forward", color='green')

plt.xlabel("episode")
plt.ylabel("# of actions x (1/100)")
plt.legend()
plt.show()

# Plot the reward
plt.figure()
ax = plt.gca()
for Learner in LEARNERS:
    for identifier in identifierList:
        name = Learner.__name__ + identifier
        data = collectorreward.getStats(name)
        plot(ax, data, label=name)

plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()

# Plot Q-values
plt.figure()
ax = plt.gca()
for Learner in LEARNERS:
    for identifier in identifierList:
        name = Learner.__name__ + identifier
        # datab = collectorbq.getStats(name)
        datas = collectorsq.getStats(name)
        # dataf = collectorfq.getStats(name)
        # plot(ax, datab, label="back", color="red")
        plot(ax, datas, label=name + " - doNothing")
        # plot(ax, dataf, label="forward", color='green')

plt.xlabel("epoch")
plt.ylabel("Q-values")
plt.legend()
plt.show()

print("Elapsed time: " + str(time.time() - start_time))
