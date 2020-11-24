from utils.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
import torch

from RlGlue import RlGlue
from agents.QLearning import QLearning
from agents.QRC import QRC
from agents.QC import QC
from agents.DQNAgent import DQN
from environments.MountainCar import MountainCar

from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper
import pandas

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

RUNS = 1
EPISODES = 10
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
            'buffer_size': 4000,
            'h1': 32,
            'h2': 32,
        })

        agent = RlGlueCompatWrapper(learner, gamma=0.99)

        # print(agent.agent.target_net.fc_out.weight)
        glue = RlGlue(agent, env)

        glue.start()
        for episode in range(EPISODES):
            glue.num_steps = 0
            glue.total_reward = 0
            glue.runEpisode(max_steps=1000)

            print(Learner.__name__, run, episode, glue.num_steps)

            collectorb.collect(
                Learner.__name__, agent.action_dict['back']/100)
            collectors.collect(
                Learner.__name__, agent.action_dict['stay']/100)
            collectorf.collect(
                Learner.__name__, agent.action_dict['forward']/100)
            collectorreward.collect(
                Learner.__name__, glue.total_reward)
            collectorbq.collect(
                Learner.__name__, sum(agent.agent.back_values)/len(agent.agent.back_values))
            collectorsq.collect(
                Learner.__name__, sum(agent.agent.stay_values)/len(agent.agent.stay_values))
            collectorfq.collect(
                Learner.__name__, sum(agent.agent.forward_values)/len(agent.agent.forward_values))

        # print(agent.agent.target_net.fc_out.weight)
        # print(len(agent.agent.back_values))
        # print(len(agent.agent.stay_values))
        # print(len(agent.agent.forward_values))
        # print(agent.agent.back_values)

        action_dict.append(agent.action_dict)
        collectorb.reset()
        collectors.reset()
        collectorf.reset()
        collectorreward.reset()
        collectorbq.reset()
        collectorsq.reset()
        collectorfq.reset()
        agent.resetDict()


df = pandas.DataFrame(action_dict)
print(action_dict)
print(df.mean())



''' plot '''
plt.figure()
ax = plt.gca()
for Learner in LEARNERS:
    name = Learner.__name__
    datab = collectorb.getStats(name)
    plot(ax, datab, label="back", color="red")

    datas = collectors.getStats(name)
    plot(ax, datas, label="do nothing", color='blue')

    dataf = collectorf.getStats(name)
    plot(ax, dataf, label="forward", color='green')
plt.xlabel("episode")
plt.ylabel("# of actions x (1/100)")
plt.legend()
plt.show()

plt.figure()
ax = plt.gca()
for Learner in LEARNERS:
    name = Learner.__name__
    data = collectorreward.getStats(name)
    plot(ax, data, label=name, color=COLORS[name])
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()

plt.figure()
ax = plt.gca()
for Learner in LEARNERS:
    name = Learner.__name__
    datab = collectorbq.getStats(name)
    plot(ax, datab, label="back", color="red")

    datas = collectorsq.getStats(name)
    plot(ax, datas, label="do nothing", color='blue')

    dataf = collectorfq.getStats(name)
    plot(ax, dataf, label="forward", color='green')
plt.xlabel("epoch")
plt.ylabel("Q-values")
plt.legend()
plt.show()
