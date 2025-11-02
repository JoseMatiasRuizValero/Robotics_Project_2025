import numpy as np
import random

class QLearningAgent:
    def __init__ (self, stateSize, actionSize, learningRate = 0.1, discountFactor = 0.95, epsilon = 1.0, epsilonMin = 0.1, epsilonDecay = 0.99):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.qTable = np.zeros((stateSize, actionSize))

    def actios(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.actionSize - 1)
        return np.argmax(self.qTable[state])
    
    def update (self, state, action, reward, nextState):
        bestNextAction = np.argmax(self.qTable[nextState])
        TDTarget = reward + self.discountFactor * self.qTable[nextState, bestNextAction]
        TDError = TDTarget - self.qTable[state, action]
        self.qTable[state, action] += self.learningRate * TDError

    def decayEpsilon(self):
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay