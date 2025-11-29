import numpy as np
import random

class QLearningAgent:
    def __init__ (self, stateSize, actionSize, learningRate = 0.1, discountFactor = 0.95, epsilon = 1.0, epsilonMin = 0.1, epsilonDecay = 0.995):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.alpha = learningRate
        self.gamma = discountFactor
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.qTable = np.zeros((stateSize, actionSize))

    def chooseAction(self, state):
        # epsilon greedy policy
        if state < 0 or state >= self.stateSize:
            state = max(0, min(self.stateSize - 1, state))
        if random.random() < self.epsilon:
            return random.randrange(self.actionSize)
        return int(np.argmax(self.qTable[state]))
    
    def update (self, state, action, reward, nextState):
        # bounds check
        if state < 0 or state >= self.stateSize:
            state = max(0, min(self.stateSize - 1, state))
        if nextState < 0 or nextState >= self.stateSize:
            nextState = max(0, min(self.stateSize - 1, nextState))
        if action < 0 or action >= self.actionSize:
            action = max(0, min(self.actionSize - 1, action))
        
        # Q-learning update
        maxNext = float(np.max(self.qTable[nextState]))
        target = reward + self.gamma * maxNext
        self.qTable[state, action] = self.qTable[state, action] + self.alpha * (target - self.qTable[state, action])

    def endEpisode(self):
        if self.epsilon > self.epsilonMin:
            self.epsilon = self.epsilon * self.epsilonDecay

    def save(self, path):
        np.save(path, self.qTable)

    def load(self, path):
        self.qTable = np.load(path)
