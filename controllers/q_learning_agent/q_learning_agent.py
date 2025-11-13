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

    def chooseAction(self, state: int)->int:
        if random.random() < self.epsilon:
            return random.randrange(self.actionSize)
        return int(np.argmax(self.qTable[state]))
    
    def update (self, state: int, action: int, reward: float, nextState: int)->None:
        bestNextAction = float(np.max(self.qTable[nextState]))
        TDTarget = reward + self.discountFactor * bestNextAction
        self.qTable[state, action] += self.learningRate * (TDTarget - float(self.qTable[state, action]))

    def endEpisode(self) -> None:
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def save(self, path: str) -> None:
        np.save(path, self.qTable)

    def load(self, path: str) -> None:
        self.qTable = np.load(path)