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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print("break")
        # print(childGameState)
        # print(newPos)
        # print(newFood)
        # print(newGhostStates)
        # print(newScaredTimes)
        "*** YOUR CODE HERE ***"
        # print(childGameState.getScore())
        # print(newFood)
        # print(sum(newFood.asList()))
        nearest_food_dist = min([util.manhattanDistance(newPos, food) for food in newFood.asList()], default=0)
        nearest_ghost_dist = max([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates], default=0)
        score = childGameState.getScore()
        if nearest_food_dist:
            score += 1 / nearest_food_dist
        if nearest_ghost_dist:
            score -= 5 / nearest_ghost_dist
        return score


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
        self.evaluationFunction = util.lookup(evalFn, globals())
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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        totalAgents = gameState.getNumAgents()
        totalActions = totalAgents * self.depth
        ''' returns a tuple of (action, reward) where state is the next state and action is the action taken to get there.
            oh yea and reward is the valuation at that location
        '''
        def helper(gameState, actionsLeft, agent):
            if gameState.isLose() or gameState.isWin():
                return (None, self.evaluationFunction(gameState))
            if actionsLeft: # not at the bottom of the tree yet.
                nextStates = [(gameState.getNextState(agent, a), a) for a in gameState.getLegalActions(agent)]
                nextHelpers = [(s[1], helper(s[0], actionsLeft - 1, (agent + 1) % totalAgents)[1]) for s in nextStates]
                if agent: # a ghost
                    return min(nextHelpers, key=lambda x: x[1], default=gameState.getLegalActions(agent)[0])
                else: # pacman's playing
                    return max(nextHelpers, key=lambda x: x[1], default=gameState.getLegalActions(agent)[0])
            else: # the bottom of the tree
                return (None, self.evaluationFunction(gameState))

        return helper(gameState, totalActions, 0)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        totalAgents = gameState.getNumAgents()
        totalActions = totalAgents * self.depth
        ''' returns a tuple of (action, reward) where state is the next state and action is the action taken to get there.
            oh yea and reward is the valuation at that location
        '''
        def helper(gameState, actionsLeft, agent, alpha, beta):
            if gameState.isLose() or gameState.isWin():
                #print('w/l')
                return (None, self.evaluationFunction(gameState))
            if actionsLeft:  # not at the bottom of the tree yet.
                best = (None, float("inf"))
                selector = lambda x: x[1]
                if not agent: # pacman
                    best = (None, float("-inf"))
                for a in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, a)
                    nextHelper = (a, helper(
                        nextState, actionsLeft - 1, (agent + 1) % totalAgents, alpha, beta)[1])
                    if agent: # a ghost
                        best = min(best, nextHelper, key=selector)
                        if best[1] < alpha[1]:
                            return best
                        beta = min(beta, best, key=selector)
                    else: # pacman
                        best = max(best, nextHelper, key=selector)
                        if best[1] > beta[1]:
                            return best
                        alpha = max(alpha, best, key=selector)
                return best
            else:  # the bottom of the tree
                return (None, self.evaluationFunction(gameState))

        return helper(gameState, totalActions, 0, (None, float('-inf')), (None, float('inf')))[0]

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
        "*** YOUR CODE HERE ***"
        totalAgents = gameState.getNumAgents()
        totalActions = totalAgents * self.depth
        ''' returns a tuple of (action, reward) where state is the next state and action is the action taken to get there.
            oh yea and reward is the valuation at that location
        '''
        def helper(gameState, actionsLeft, agent):
            if gameState.isLose() or gameState.isWin():
                #print('w/l')
                return (None, self.evaluationFunction(gameState))
            if actionsLeft:  # not at the bottom of the tree yet.
                best = (None, 0)
                def selector(x): return x[1]
                if not agent:  # pacman
                    best = (None, float("-inf"))
                for a in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, a)
                    nextHelper = (a, helper(
                        nextState, actionsLeft - 1, (agent + 1) % totalAgents)[1])
                    if agent:  # a ghost
                        best = (a, nextHelper[1] / len(gameState.getLegalActions(agent)) + best[1])
                    else:  # pacman
                        best = max(best, nextHelper, key=selector)
                return best
            else:  # the bottom of the tree
                return (None, self.evaluationFunction(gameState))

        return helper(gameState, totalActions, 0)[0]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    childGameState = currentGameState
    newPos = childGameState.getPacmanPosition()
    newFood = childGameState.getFood()
    newGhostStates = childGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # print("break")
    # print(childGameState)
    # print(newPos)
    # print(newFood)
    # print(newGhostStates)
    "*** YOUR CODE HERE ***"
    # print(childGameState.getScore())
    # print(newFood)
    # print(sum(newFood.asList()))
    nearest_food_dist = min([util.manhattanDistance(newPos, food) for food in newFood.asList()], default=0)
    nearest_ghost_dist = min([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates], default=0)
    score = childGameState.getScore()
    if nearest_food_dist:
        score += 1 / nearest_food_dist
    else:
        score += 1
    if nearest_ghost_dist:
        score -= 1 / nearest_ghost_dist
    if len(newFood.asList()) != 0:
        score -= 1 / len(newFood.asList())
    score += sum(newScaredTimes)
    return score

# Abbreviation
better = betterEvaluationFunction
