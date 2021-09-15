# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print(scores)
        print(legalMoves)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        dis = 100
        foodPoints = 0

        for pos in newFood.asList():
          d = manhattanDistance(pos,newPos)
          d+=1
          foodPoints += 1.0/(d*d)
          
        for ghostState in newGhostStates:
          if manhattanDistance(newPos,ghostState.getPosition()) < 2:
            dis = 0

        print(foodPoints)
        return  dis + (foodPoints * 16) + successorGameState.getScore() 

        
        


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = betterEvaluationFunction
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        n = gameState.getNumAgents()
        scoreList = []

        def recMiniMax(state, i):
          if i >= n*self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          
          if i%n != 0: 
            score = float('inf')
            for action in state.getLegalActions(i%n):
              nextState = state.generateSuccessor(i%n,action)
              score = min(score, recMiniMax(nextState, i+1))
            return score

          else: 
            score = -float('inf')
            for action in state.getLegalActions(i%n):
              newState = state.generateSuccessor(i%n,action)
              score = max(score, recMiniMax(newState, i+1))
              if i == 0:
                scoreList.append(score)
            return score
          
        recMiniMax(gameState, 0)
        return gameState.getLegalActions(0)[scoreList.index(max(scoreList))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        n = gameState.getNumAgents()
        scoreList = []
        
        def recAlphaBeta(state, i, alpha, beta):
          if i >= n*self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          
          if i%n != 0: 
            score = float('inf')
            for action in state.getLegalActions(i%n):
              nextState = state.generateSuccessor(i%n,action)
              score = min(score, recAlphaBeta(nextState, i+1, alpha, beta))
              beta = min(beta,score)
              if beta < alpha:
                return score
            return score

          else: 
            score = -float('inf')
            for action in state.getLegalActions(i%n):
              newState = state.generateSuccessor(i%n,action)
              score = max(score, recAlphaBeta(newState, i+1, alpha, beta))
              if i == 0:
                scoreList.append(score)
              alpha = max(alpha, score)
              if alpha > beta:
                return score
            return score
          
        recAlphaBeta(gameState, 0, -float('inf'), float('inf'))
        return gameState.getLegalActions(0)[scoreList.index(max(scoreList))]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        n = gameState.getNumAgents()
        scoreList = []

        def recExpectiMax(state, i):
          if i >= n*self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          
          if i%n != 0: 
            score = 0.0
            den = 0.0
            for action in state.getLegalActions(i%n):
              nextState = state.generateSuccessor(i%n,action)
              score += recExpectiMax(nextState, i+1)
              den += 1.0
            return score/den

          else: 
            score = -float('inf')
            for action in state.getLegalActions(i%n):
              newState = state.generateSuccessor(i%n,action)
              score = max(score, recExpectiMax(newState, i+1))
              if i == 0:
                scoreList.append(score)
            return score
          
        recExpectiMax(gameState, 0)
        return gameState.getLegalActions(0)[scoreList.index(max(scoreList))]
        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        
    dis = 100
    foodPoints = 0

    for pos in newFood.asList():
      d = manhattanDistance(pos,newPos)
      d+=1
      foodPoints += 1.0/(d*d)
          
    for i in range(len(newGhostStates)):
      if manhattanDistance(newPos,newGhostStates[i].getPosition()) < 2 and newScaredTimes[i]==0:
        dis = 0
      elif newScaredTimes[i] > 0:
        dis+=100

    
    if(newScaredTimes[0]>0):
      dis += 50
    return  dis + (foodPoints * 16) + currentGameState.getScore() 

# Abbreviation
better = betterEvaluationFunction

