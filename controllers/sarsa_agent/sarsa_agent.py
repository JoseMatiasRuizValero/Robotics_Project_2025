import numpy as np
import random

class SARSAAgent:
    def __init__ (self, stateSize, actionSize, learningRate = 0.1, discountFactor = 0.95, epsilon = 1.0, epsilonMin = 0.1, epsilonDecay = 0.99):
        
        #   Assign vales
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        #   Initialize qTable to 0
        self.qTable = np.zeros((stateSize, actionSize))

    def chooseAction(self, state: int)->int:
        #   Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randrange(self.actionSize)
        return int(np.argmax(self.q[state]))
    
    def update (self, state: int, action: int, reward: float, nextState: int)->None:
        # SARSA update (on policy)
        currentQ = float(self.qTable[state, action])
        nextQ = float(self.qTable[nextState, nextAction])
        TDTarget = reward + self.discountFactor * nextQ
        TDError = TDTarget - currentQ
        self.qTable[state, action] += self.learningRate * TDError

    def endEpisode(self) -> None:
        # Epsilon Decay
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def save(self, path: str) -> None:
        np.save(path, self.qTable)

    def load(self, path: str) -> None:
        self.qTable = np.load(path)