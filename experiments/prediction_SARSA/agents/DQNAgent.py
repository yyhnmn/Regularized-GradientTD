import torch
import numpy as np
import torch.nn.functional as f
from agents.BaseAgent import BaseAgent
from utils.torch import device, getBatchColumns
from utils.ReplayBuffer import ReplayBuffer
from agents.Network import Network
from utils.torch import device


def choice(arr, size=1):
    idxs = np.random.permutation(len(arr))
    return [arr[i] for i in idxs[:size]]


class DQN(BaseAgent):
    def __init__(self, features, actions, state_array, params):
        super(DQN, self).__init__(features, actions , params)
        self.buffer_BACK = ReplayBuffer(1000)
        self.buffer_STAY = ReplayBuffer(1000)
        self.buffer_FORWARD = ReplayBuffer(1000)
        self.back_values = []
        self.stay_values = []
        self.forward_values = []
        self.td_loss = []
        self.state_array = state_array

    def updateNetwork(self, samples):
        # organize the mini-batch so that we can request "columns" from the data
        # e.g. we can get all of the actions, or all of the states with a single call
        batch = getBatchColumns(samples)

        # compute Q(s, a) for each sample in mini-batch
        Qs, x = self.policy_net(batch.states)
        Qsa = Qs.gather(1, batch.actions).squeeze()

        # by default Q(s', a') = 0 unless the next states are non-terminal
        Qspap = torch.zeros(batch.size, device=device)
        
        # if we don't have any non-terminal next states, then no need to bootstrap
        if batch.nterm_sp.shape[0] > 0:
            Qsp, _ = self.target_net(batch.nterm_sp)
            
            nterm_action = self.selectAction(batch.nterm_sp)
            # Qsp, _ = self.target_net(batch.nterm_sp, batch.nterm_action)

            # bootstrapping term is the max Q value for the next-state
            # only assign to indices where the next state is non-terminal
            Qspap[batch.nterm] = Qsp[torch.arange(Qsp.size(0)), nterm_action]

        # compute the empirical MSBE for this mini-batch and let torch auto-diff to optimize
        # don't worry about detaching the bootstrapping term for semi-gradient Q-learning
        # the target network handles that
        target = batch.rewards + batch.gamma * Qspap
        td_loss = 0.5 * f.mse_loss(target, Qsa)

        # make sure we have no gradients left over from previous update
        self.optimizer.zero_grad()
        # self.target_net.zero_grad()

        # compute the entire gradient of the network using only the td error
        td_loss.backward()
        self.td_loss.append(td_loss.detach().numpy())
        
        # self.td_loss = self.td_loss + list(td_loss.detach().numpy())
        
        
        Qs_state_array, _ = self.policy_net(self.state_array)
        
        Qsa_mean_states = torch.mean(Qs_state_array,0)
        
        self.back_values.append(Qsa_mean_states[0].detach().numpy())
        self.stay_values.append(Qsa_mean_states[1].detach().numpy())
        self.forward_values.append(Qsa_mean_states[2].detach().numpy())
   
        
        # update the *policy network* using the combined gradients
        self.optimizer.step()
        
        
    def update(self, s, a, sp, r, gamma):
        # the "online" sample gets tossed into the replay buffer
        self.buffer.add((s, a, sp, r, gamma))
        self.steps += 1

        # if it is time to set the target net <- policy network
        # do that before the learning step
        if self.steps % self.target_refresh == 0:
            self.policy_net.cloneWeightsTo(self.target_net)

        # as long as we have enough samples in the buffer to do one mini-batch update
        # go ahead and randomly sample a mini-batch and do a single update
        if len(self.buffer) > 64:
            samples, idcs = self.buffer.sample(64)
            self.updateNetwork(samples)



    # def update(self, s, a, sp, r, gamma):
    #     if a.cpu().numpy() == 0:
    #         self.buffer_BACK.add((s, a, sp, r, gamma))
    #     elif a.cpu().numpy() == 1:
    #         self.buffer_STAY.add((s, a, sp, r, gamma))
    #     elif a.cpu().numpy() == 2:
    #         self.buffer_FORWARD.add((s, a, sp, r, gamma))

    #     wholebuffer = self.buffer_BACK.buffer + \
    #         self.buffer_STAY.buffer+self.buffer_FORWARD.buffer

    #     # the "online" sample gets tossed into the replay buffer
    #     self.buffer.add((s, a, sp, r, gamma))
    #     self.steps += 1

    #     # if it is time to set the target net <- policy network
    #     # do that before the learning step
    #     if self.steps % self.target_refresh == 0:
    #         self.policy_net.cloneWeightsTo(self.target_net)

    #     # as long as we have enough samples in the buffer to do one mini-batch update
    #     # go ahead and randomly sample a mini-batch and do a single update
    #     if len(self.buffer) > 32:
    #         samples = choice(wholebuffer, 32)
    #         # samples, idcs = self.buffer.sample(32)
    #         self.updateNetwork(samples)
