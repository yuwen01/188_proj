import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        h1 = 300
        h2 = 300
        self.set_weights([nn.Parameter(state_dim, h1), nn.Parameter(1, h1), nn.Parameter(h1, h2), 
            nn.Parameter(1, h2), nn.Parameter(h2, action_dim), nn.Parameter(1, action_dim)])
        self.learning_rate = -1
        self.numTrainingGames = 3000
        self.batch_size = 300

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        w1 = self.parameters[0]
        b1 = self.parameters[1]
        w2 = self.parameters[2]
        b2 = self.parameters[3]
        w3 = self.parameters[4]
        b3 = self.parameters[5]

        first = nn.ReLU(nn.AddBias(nn.Linear(states, w1), b1))
        second = nn.ReLU(nn.AddBias(nn.Linear(first, w2), b2))
        return nn.AddBias(nn.Linear(second, w3), b3)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        my_gradients = nn.gradients(self.get_loss(states, Q_target), self.parameters)
        for i in range(len(self.parameters)):
            self.parameters[i].update(my_gradients[i], self.learning_rate)
