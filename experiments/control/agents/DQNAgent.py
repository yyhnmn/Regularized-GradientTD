import torch
import numpy as np
import torch.nn.functional as f
import math

from agents.BaseAgent import BaseAgent
from utils.torch import device, getBatchColumns
from utils.ReplayBuffer import ReplayBuffer
from agents.Network import Network
from utils.torch import device


def choice(arr, size=1):
    idxs = np.random.permutation(len(arr))
    return [arr[i] for i in idxs[:size]]


class DQN(BaseAgent):
    def __init__(self, features, actions, params):
        super(DQN, self).__init__(features, actions, params)
        self.buffer_BACK = ReplayBuffer(1000)
        self.buffer_STAY = ReplayBuffer(1000)
        self.buffer_FORWARD = ReplayBuffer(1000)
        self.back_values = []
        self.stay_values = []
        self.forward_values = []
        self.ratioMap = params['ratioMap']
        self.sampleSize = params['sampleSize']

    def updateNetwork(self, samples):
        # organize the mini-batch so that we can request "columns" from the data
        # e.g. we can get all of the actions, or all of the states with a single call
        batch = getBatchColumns(samples)

        # compute Q(s, a) for each sample in mini-batch
        Qs, x = self.policy_net(batch.states)
        Qsa = Qs.gather(1, batch.actions).squeeze()

        # by default Q(s', a') = 0 unless the next states are non-terminal

        Qspap = torch.zeros(batch.size, device=device)
        for i in range(30):
            if batch.actions.numpy()[i][0] == 0:
                self.back_values.append(Qsa.detach().numpy()[i])
            elif batch.actions.numpy()[i][0] == 1:
                self.stay_values.append(Qsa.detach().numpy()[i])
            elif batch.actions.numpy()[i][0] == 2:
                self.forward_values.append(Qsa.detach().numpy()[i])

        # if we don't have any non-terminal next states, then no need to bootstrap
        if batch.nterm_sp.shape[0] > 0:
            Qsp, _ = self.target_net(batch.nterm_sp)

            # bootstrapping term is the max Q value for the next-state
            # only assign to indices where the next state is non-terminal
            Qspap[batch.nterm] = Qsp.max(1).values

        # compute the empirical MSBE for this mini-batch and let torch auto-diff to optimize
        # don't worry about detaching the bootstrapping term for semi-gradient Q-learning
        # the target network handles that
        target = batch.rewards + batch.gamma * Qspap.detach()
        td_loss = 0.5 * f.mse_loss(target, Qsa)

        # make sure we have no gradients left over from previous update
        self.optimizer.zero_grad()
        self.target_net.zero_grad()

        # compute the entire gradient of the network using only the td error
        td_loss.backward()

        # update the *policy network* using the combined gradients
        self.optimizer.step()

    def update(self, s, a, sp, r, gamma):
        if a.cpu().numpy() == 0:
            self.buffer_BACK.add((s, a, sp, r, gamma))
        elif a.cpu().numpy() == 1:
            self.buffer_STAY.add((s, a, sp, r, gamma))
        elif a.cpu().numpy() == 2:
            self.buffer_FORWARD.add((s, a, sp, r, gamma))

        # the "online" sample gets tossed into the replay buffer
        self.buffer.add((s, a, sp, r, gamma))
        self.steps += 1

        # if it is time to set the target net <- policy network
        # do that before the learning step
        if self.steps % self.target_refresh == 0:
            self.policy_net.cloneWeightsTo(self.target_net)

        back_sample_count = math.floor(self.ratioMap.backward_ratio * self.sampleSize)
        stay_sample_count = math.floor(self.ratioMap.stay_ratio * self.sampleSize)
        forward_sample_count = math.floor(self.ratioMap.forward_ratio * self.sampleSize)

        # as long as we have enough samples in the buffer to do one mini-batch update
        # go ahead and randomly sample a mini-batch and do a single update
        if len(self.buffer_BACK) > back_sample_count \
                and len(self.buffer_STAY) > stay_sample_count \
                and len(self.buffer_FORWARD) > forward_sample_count:
            samplesBack, idcs = self.buffer_BACK.sample(back_sample_count)
            samplesStay, idcs = self.buffer_STAY.sample(stay_sample_count)
            samplesForward, idcs = self.buffer_FORWARD.sample(forward_sample_count)
            samples = samplesBack + samplesStay + samplesForward
            self.updateNetwork(samples)
