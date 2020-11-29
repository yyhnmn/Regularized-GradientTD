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

RUNS = 10
EPISODES = 3
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
    'DQN': 0.0002765,
}

P = np.random.uniform(low=-1.2, high=0.5, size=1000)
V = np.random.uniform(low=-0.07, high=0.07, size=1000)
S_ARR = s = np.transpose(np.vstack((P, V)))
S_ARR = torch.from_numpy(S_ARR).to(torch.float32)

collectorb = Collector()
collectors = Collector()
collectorf = Collector()
collectorreward = Collector()
collectorbq = Collector()
collectorsq = Collector()
collectorfq = Collector()
collectorloss = Collector()


for run in range(RUNS):
    for Learner in LEARNERS:
        np.random.seed(run)
        torch.manual_seed(run)

        env = MountainCar()

        learner = Learner(env.features, env.num_actions,S_ARR, {
            'alpha': STEPSIZES[Learner.__name__],
            'epsilon': 0.2,
            'beta': 1.0,
            'target_refresh': 1,
            'buffer_size': 4000,
            'h1': 32,
            'h2': 32,
        })

        agent = RlGlueCompatWrapper(learner, gamma=0.5)

        # print(agent.agent.target_net.fc_out.weight)
        glue = RlGlue(agent, env)

        glue.start()
        for episode in range(EPISODES):
            glue.num_steps = 0
            glue.total_reward = 0
            glue.runEpisode(max_steps=6000)

            print(Learner.__name__, run, episode, glue.num_steps)
            
            collectorreward.collect(
                Learner.__name__, glue.total_reward)

            # collectorb.collect(
            #     Learner.__name__, agent.action_dict['back']/100)
            # collectors.collect(
            #     Learner.__name__, agent.action_dict['stay']/100)
            # collectorf.collect(
            #     Learner.__name__, agent.action_dict['forward']/100)
            
            
            # if len(agent.agent.back_values) == 0:
            #     collectorsq.collect(Learner.__name__, 0)
            # else:
            #     collectorbq.collect(
            #     Learner.__name__, sum(agent.agent.back_values)/len(agent.agent.back_values))
            
            # if len(agent.agent.stay_values) == 0:
            #     collectorsq.collect(Learner.__name__, 0)
            # else:
            #     collectorsq.collect(
            #     Learner.__name__, sum(agent.agent.stay_values)/len(agent.agent.stay_values))
            
            # if len(agent.agent.forward_values) == 0:
            #     collectorsq.collect(Learner.__name__, 0)
            # else:                
            #     collectorfq.collect(
            #     Learner.__name__, sum(agent.agent.forward_values)/len(agent.agent.forward_values))
                
        collectorloss.collect_list(Learner.__name__, agent.agent.td_loss)
        collectorbq.collect_list(Learner.__name__, agent.agent.back_values)
        collectorsq.collect_list(Learner.__name__, agent.agent.stay_values)
        collectorfq.collect_list(Learner.__name__, agent.agent.forward_values)


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
        collectorloss.reset()
        agent.resetDict()


df = pandas.DataFrame(action_dict)
print(action_dict)
print(df.mean())



''' plot '''
plt.figure()
ax = plt.gca()
for Learner in LEARNERS:
    name = Learner.__name__
    datab = collectorloss.getStats_list(name)
    plot(ax, datab, label="Loss", color="red")

plt.xlabel("Update Steps")
plt.ylabel("TD Loss")
plt.legend()
plt.show()

# plt.figure()
# ax = plt.gca()
# for Learner in LEARNERS:
#     name = Learner.__name__
#     datab = collectorb.getStats(name)
#     plot(ax, datab, label="back", color="red")

#     datas = collectors.getStats(name)
#     plot(ax, datas, label="do nothing", color='blue')

#     dataf = collectorf.getStats(name)
#     plot(ax, dataf, label="forward", color='green')
# plt.xlabel("episode")
# plt.ylabel("# of actions x (1/1000)")
# plt.legend()
# plt.show()

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
    datab = collectorbq.getStats_list(name)
    plot(ax, datab, label="back", color="red")

    datas = collectorsq.getStats_list(name)
    plot(ax, datas, label="do nothing", color='blue')

    dataf = collectorfq.getStats_list(name)
    plot(ax, dataf, label="forward", color='green')
plt.xlabel("Update steps")
plt.ylabel("Q-values")
plt.legend()
plt.show()
