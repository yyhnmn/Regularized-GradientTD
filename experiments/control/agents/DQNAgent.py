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
    def __init__(self, features, actions,  state_array, params):
        super(DQN, self).__init__(features, actions, params)
        self.buffer_BACK = ReplayBuffer(1000)
        self.buffer_STAY = ReplayBuffer(1000)
        self.buffer_FORWARD = ReplayBuffer(1000)
        
        self.back_q_net = Network(features, self.h1, self.h2, 1).to(device)
        self.back_target_q_net = Network(
            features, self.h1, self.h2, 1).to(device)
        self.back_q_net.cloneWeightsTo(self.back_target_q_net)

        self.stay_q_net = Network(features, self.h1, self.h2, 1).to(device)
        self.stay_target_q_net = Network(
            features, self.h1, self.h2, 1).to(device)
        self.stay_q_net.cloneWeightsTo(self.stay_target_q_net)

        self.forward_q_net = Network(features, self.h1, self.h2, 1).to(device)
        self.forward_target_q_net = Network(
            features, self.h1, self.h2, 1).to(device)
        self.forward_q_net.cloneWeightsTo(self.forward_target_q_net)
        
        self.optimizerBack = optim.Adam(self.back_q_net.parameters(), lr=self.alpha, betas=(0.9, 0.999))
        self.optimizerStay = optim.Adam(self.stay_q_net.parameters(), lr=self.alpha, betas=(0.9, 0.999))
        self.optimizerForward = optim.Adam(self.forward_q_net.parameters(), lr=self.alpha, betas=(0.9, 0.999))
        
        
        self.back_values = []
        self.stay_values = []
        self.forward_values = []
        
        self.back_values_baseline = []
        self.stay_values_baseline = []
        self.forward_values_baseline = []
        
        self.td_loss = []
        self.state_array = state_array
        
        
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
        # for i in range(len(batch.actions.numpy())):
        #     if batch.actions.numpy()[i][0] == 0:
        #         self.back_values.append(Qsa.detach().numpy()[i])
        #     elif batch.actions.numpy()[i][0] == 1:
        #         self.stay_values.append(Qsa.detach().numpy()[i])
        #     elif batch.actions.numpy()[i][0] == 2:
        #         self.forward_values.append(Qsa.detach().numpy()[i])

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
        
        self.td_loss.append(td_loss.detach().numpy())
        
        # self.td_loss = self.td_loss + list(td_loss.detach().numpy())
        
        
        Qs_state_array, _ = self.policy_net(self.state_array)
        
        Qsa_mean_states = torch.mean(Qs_state_array,0)
        
        self.back_values.append(Qsa_mean_states[0].detach().numpy())
        self.stay_values.append(Qsa_mean_states[1].detach().numpy())
        self.forward_values.append(Qsa_mean_states[2].detach().numpy())

        # update the *policy network* using the combined gradients
        self.optimizer.step()
        
    def updateActionNet(self, samples, q_net, target_q_net,optimizer,storeList):
        batch = getBatchColumns(samples)
        Qs, x = q_net(batch.states)

        # Qsa = Qs.squeeze()
        # for i in range(len(batch.actions)):
        #     storeList.append(Qsa.detach().numpy()[i])
        Qspap = torch.zeros(batch.size, device=device)
        
        ############  ============  CHECK ================= ###############################
        if batch.nterm_sp.shape[0] > 0:
            
            ##  Qsp, _ = target_q_net(batch.nterm_sp) #### Is this correct ????
            
            Qsp_back,_ = self.back_target_q_net(batch.nterm_sp)
            Qsp_stay,_ = self.stay_target_q_net(batch.nterm_sp)
            Qsp_forward,_ = self.forward_target_q_net(batch.nterm_sp)
            
            Qsp = torch.hstack([Qsp_back,Qsp_stay,Qsp_forward])

            # bootstrapping term is the max Q value for the next-state
            # only assign to indices where the next state is non-terminal
            Qspap[batch.nterm] = Qsp.max(1).values

        ############  ============  CHECK ================= ###############################
        # compute the empirical MSBE for this mini-batch and let torch auto-diff to optimize
        # don't worry about detaching the bootstrapping term for semi-gradient Q-learning
        # the target network handles that
        target = batch.rewards + batch.gamma * Qspap.detach()
        td_loss = 0.5 * f.mse_loss(target, Qsa)

        # make sure we have no gradients left over from previous update
        optimizer.zero_grad()
        target_q_net.zero_grad()
        self.back_target_q_net.zero_grad()
        self.stay_target_q_net.zero_grad()
        self.forward_target_q_net.zero_grad()

        # compute the entire gradient of the network using only the td error
        td_loss.backward()
        
        Qs_state_array, _ = q_net(self.state_array)
        Qsa_mean_states = torch.mean(Qs_state_array,0)
        storeList.append(Qsa_mean_states[0].detach().numpy())

        # update the *policy network* using the combined gradients
        optimizer.step()

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
            self.back_q_net.cloneWeightsTo(self.back_target_q_net)
            self.stay_q_net.cloneWeightsTo(self.stay_target_q_net)
            self.forward_q_net.cloneWeightsTo(self.forward_target_q_net)


        back_sample_count = math.floor(
            self.ratioMap.backward_ratio * self.sampleSize)
        stay_sample_count = math.floor(
            self.ratioMap.stay_ratio * self.sampleSize)
        forward_sample_count = math.floor(
            self.ratioMap.forward_ratio * self.sampleSize)

        # as long as we have enough samples in the buffer to do one mini-batch update
        # go ahead and randomly sample a mini-batch and do a single update
        if len(self.buffer_BACK) > back_sample_count \
                and len(self.buffer_STAY) > stay_sample_count \
                and len(self.buffer_FORWARD) > forward_sample_count:
            samplesBack, idcs = self.buffer_BACK.sample(back_sample_count)
            samplesStay, idcs = self.buffer_STAY.sample(stay_sample_count)
            samplesForward, idcs = self.buffer_FORWARD.sample(
                forward_sample_count)
            self.updateActionNet(samplesBack,self.back_q_net,self.back_target_q_net,self.optimizerBack,self.back_values_baseline)  
            self.updateActionNet(samplesStay,self.stay_q_net,self.stay_target_q_net,self.optimizerStay,self.stay_values_baseline)
            self.updateActionNet(samplesForward,self.forward_q_net,self.forward_target_q_net,self.optimizerForward,self.forward_values_baseline)
            samples = samplesBack + samplesStay + samplesForward
            
            self.updateNetwork(samples)


            
            
    
