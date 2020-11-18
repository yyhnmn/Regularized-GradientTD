from RlGlue import BaseAgent
import torch
from utils.torch import device, toNP


# stores the previous state for 1-step bootstrapping methods (all methods in this project)
# casts all observations into torch tensors before storing so that we don't cast the same data multiple times (really important when running on GPU)
# this wrapper allows easy API compatibility with RLGlue, while allowing the learning code to remain small and simple (the complexity due to bookkeeping is pushed into this class)
class RlGlueCompatWrapper(BaseAgent):
    def __init__(self, agent, gamma):
        self.agent = agent
        self.gamma = gamma
        self.s = None
        self.x = None
        self.a = None
        self.action_dict = {'back':0,'stay':0,'forward':0}
        self.action_dict_proportion = {'back':0,'stay':0,'forward':0}

    def resetDict(self):
        self.action_dict=self.action_dict.fromkeys(self.action_dict,0)
        self.action_dict_proportion=self.action_dict_proportion.fromkeys(self.action_dict_proportion,0)

    # called on the first step of the episode
    def start(self, s):
        self.s = s
        self.x = torch.tensor(s, device=device).unsqueeze(0)
        self.a = self.agent.selectAction(self.x)
        return toNP(self.a)

    # called on all subsequent steps of the episode except the terminal step
    def step(self, r, sp):

        xp = torch.tensor(sp, device=device).unsqueeze(0)
        r = torch.tensor(r, device=device).unsqueeze(0)
        self.agent.update(self.x, self.a, r, xp, self.gamma)
        a = self.a.numpy()
        if a == 0:
            self.action_dict['back'] += 1
        elif a == 1:
            self.action_dict['stay'] += 1
        elif a == 2:
            self.action_dict['forward'] += 1

        sum_action = 0
        for (key, value) in self.action_dict.items(): 
            sum_action += value
        self.action_dict_proportion['back'] = (self.action_dict['back'] / (sum_action + 0.0))*100
        self.action_dict_proportion['stay'] = (self.action_dict['stay'] / (sum_action + 0.0))*100
        self.action_dict_proportion['forward'] = (self.action_dict['forward'] / (sum_action + 0.0))*100
        
        self.s = sp
        self.x = xp
        self.a = self.agent.selectAction(xp)

        # make sure the environment is not operating on a pytorch tensor
        return toNP(self.a)

    # called on the terminal step of the episode
    def end(self, r):
        r = torch.tensor(r, device=device).unsqueeze(0)
        self.agent.update(self.x, self.a, r, None, 0)
