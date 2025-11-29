import numpy as np
import random

class SARSAAgent:
    def __init__ (self, stateSize, actionSize, learningRate = 0.1, discountFactor = 0.95, epsilon = 1.0, epsilonMin = 0.1, epsilonDecay = 0.995):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.alpha = learningRate
        self.gamma = discountFactor
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.qTable = np.zeros((stateSize, actionSize))

    def chooseAction(self, state: int)->int:
        # epsilon greedy
        if state < 0 or state >= self.stateSize:
            state = max(0, min(self.stateSize - 1, state))
        if random.random() < self.epsilon:
            return random.randrange(self.actionSize)
        return int(np.argmax(self.qTable[state]))
    
    def update (self, state, action, reward, nextState, nextAction):
        # SARSA update
        if state < 0 or state >= self.stateSize:
            state = max(0, min(self.stateSize - 1, state))
        if nextState < 0 or nextState >= self.stateSize:
            nextState = max(0, min(self.stateSize - 1, nextState))
        if action < 0 or action >= self.actionSize:
            action = max(0, min(self.actionSize - 1, action))
        if nextAction < 0 or nextAction >= self.actionSize:
            nextAction = max(0, min(self.actionSize - 1, nextAction))
        
        currentQ = float(self.qTable[state, action])
        nextQ = float(self.qTable[nextState, nextAction])
        target = reward + self.gamma * nextQ
        error = target - currentQ
        self.qTable[state, action] = self.qTable[state, action] + self.alpha * error

    def endEpisode(self):
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def save(self, path: str):
        np.save(path, self.qTable)

    def load(self, path: str):
        self.qTable = np.load(path)

