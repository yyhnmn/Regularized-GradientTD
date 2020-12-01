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
from utils.ratiomap import RatioMap
from utils.serializermodule import SerializerModule

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
collectorbq_baseline = Collector()
collectorsq_baseline = Collector()
collectorfq_baseline = Collector()
collectorloss = Collector()
collector_penultimate_features = Collector()


def run_for_ratio(ratio_map, identifier):
    for run in range(RUNS):
        for Learner in LEARNERS:
            np.random.seed(run)
            serializer = SerializerModule('../temp/', str(run) + "-" + identifier)

            torch.manual_seed(run)

            env = MountainCar()

            learner = Learner(env.features, env.num_actions, S_ARR, {
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
                
                collectorreward.collect(
                    Learner.__name__ + identifier, glue.total_reward)

                collectorb.collect(
                    Learner.__name__ + identifier, agent.action_dict['back'] / 100)
                collectors.collect(
                    Learner.__name__ + identifier, agent.action_dict['stay'] / 100)
                collectorf.collect(
                    Learner.__name__ + identifier, agent.action_dict['forward'] / 100)
                
                # collectorbq.collect(
                #     Learner.__name__ + identifier, sum(agent.agent.back_values) / len(agent.agent.back_values))
                # collectorsq.collect(
                #     Learner.__name__ + identifier, sum(agent.agent.stay_values) / len(agent.agent.stay_values))
                # collectorfq.collect(
                #     Learner.__name__ + identifier, sum(agent.agent.forward_values) / len(agent.agent.forward_values))
                
            collectorloss.collect_list(Learner.__name__, agent.agent.td_loss)
            collector_penultimate_features.collect_list(Learner.__name__, agent.agent.penultimate_features)
            collectorbq.collect_list(Learner.__name__, agent.agent.back_values)
            collectorsq.collect_list(Learner.__name__, agent.agent.stay_values)
            collectorfq.collect_list(Learner.__name__, agent.agent.forward_values)
            
            collectorbq_baseline.collect_list(Learner.__name__, agent.agent.back_values)
            collectorsq_baseline.collect_list(Learner.__name__, agent.agent.stay_values)
            collectorfq_baseline.collect_list(Learner.__name__, agent.agent.forward_values)

            action_dict.append(agent.action_dict)
            collectorb.reset()
            collectors.reset()
            collectorf.reset()
            collectorreward.reset()
            collectorbq.reset()
            collectorsq.reset()
            collectorfq.reset()
            collectorbq_baseline.reset()
            collectorsq_baseline.reset()
            collectorfq_baseline.reset()
            collectorloss.reset()
            collector_penultimate_features.reset()

            serializer.add_to_serializer(SerializerModule.COLLECTOR_B, collectorb)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_S, collectors)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_F, collectorf)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_REWARD, collectorreward)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_BQ, collectorbq)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_SQ, collectorsq)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_FQ, collectorfq)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_BQ_BASELINE, collectorbq_baseline)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_SQ_BASELINE, collectorsq_baseline)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_FQ_BASELINE, collectorfq_baseline)
            serializer.add_to_serializer(SerializerModule.COLLECTOR_LOSS, collectorloss)
            serializer.add_to_serializer(SerializerModule.PENULTIMATE_FEATURES, collector_penultimate_features)

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


## PLOT FUNCTIONALITY REMAINS

"""
## DHRUV, YOU SHOULD SAVE :
            collectorb
            collectors
            collectorf
            collectorreward
            collectorbq
            collectorsq
            collectorfq
            collectorbq_baseline
            collectorsq_baseline
            collectorfq_baseline
            collectorloss
            also net params (last layer)

"""

print("Elapsed time: " + str(time.time() - start_time))
