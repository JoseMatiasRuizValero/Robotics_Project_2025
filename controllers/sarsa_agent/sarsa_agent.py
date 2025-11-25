import numpy as np
import random

class SARSAAgent:
    def __init__ (self, stateSize, actionSize, learningRate = 0.7, discountFactor = 0.90, epsilon = 1.0, epsilonMin = 0.1, epsilonDecay = 0.995):
        
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
        # ensure state is within valid range
        if state < 0 or state >= self.stateSize:
            state = max(0, min(self.stateSize - 1, state))
        if random.random() < self.epsilon:
            return random.randrange(self.actionSize)
        return int(np.argmax(self.qTable[state]))
    
    def update (self, state: int, action: int, reward: float, nextState: int, nextAction: int)->None:
        # SARSA update (on policy)
        # ensure state, nextState, action, nextAction are within valid range
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

