import numpy as np
import torch
import torch.nn.functional as f
import torch.optim as optim

from agents.Network import Network
from utils.ReplayBuffer import ReplayBuffer
from utils.torch import device

class BaseAgent:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.target_refresh = params['target_refresh']
        self.buffer_size = params['buffer_size']

        self.h1 = params['h1']
        self.h2 = params['h2']

        # build two networks, one for the "online" learning policy
        # the other as a fixed target network
        self.policy_net = Network(features, self.h1, self.h2, actions).to(device)
        self.target_net = Network(features, self.h1, self.h2, actions).to(device)
        self.det_net = Network(features, self.h1, self.h2, actions).to(device)
        self.bpolicy_net = Network(features, self.h1, self.h2, actions).to(device)
        self.bpolicy_net.load_state_dict(torch.load("/home/soumyadeep/Action_Imbalance/RLGTD/experiments/prediction_SARSA/agents/net_params.pt"))


        # build the optimizer for _only_ the policy network
        # target network parameters will be copied from the policy net periodically
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha, betas=(0.9, 0.999))

        # a simple circular replay buffer (i.e. a FIFO buffer)
        self.buffer = ReplayBuffer(self.buffer_size)
        self.steps = 0

        # initialize the weights of the target network to match the weights of policy network
        self.policy_net.cloneWeightsTo(self.target_net)

    def selectAction(self, x):
        # take a random action about epsilon percent of the time
        q_s, _ = self.bpolicy_net(x)
        
        if q_s.shape[0] == 3:
            q_s = q_s.unsqueeze(0)
            #act = q_s.argmax().detach()
       # else:
        act = torch.max(q_s,1).indices.detach().numpy()
            
    
        for i in range(act.shape[0]):
            action = act[i]
            if action == 1:
                if np.random.rand() < self.epsilon:
                    act[i] = np.random.choice([0, 2])
                    
        
        # if act.cpu().numpy() == 1:
        #     if np.random.rand() < self.epsilon:
        #         a = np.random.randint(self.actions-1)
                
                
        # if np.random.rand() < self.epsilon:
        #     a = np.random.randint(self.actions)
        #     return torch.tensor(a, device=device)

        # # otherwise take a greedy action
        # q_s, _ = self.bpolicy_net(x)
        # # print(q_s)
        # return q_s.argmax().detach()
        act_tensor = torch.from_numpy(act).detach().to(device)
    
        return act_tensor

    def updateNetwork(self, samples):
        pass

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
        if len(self.buffer) > 200:
            samples, idcs = self.buffer.sample(200)
            self.updateNetwork(samples)
