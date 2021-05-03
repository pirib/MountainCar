from math import pi, cos
import numpy as np
from random import random, choice
import matplotlib.pyplot as plt

import tensorflow as tf


# Helpers
def argmax(l, f = lambda e : e, *args):
    b = l[0]
    for e in l[1:]:
        if f(e, *args) > f(b, *args):
            b = e
    return b


# The environment class
class Env():
     
    # Initialize the environment according to the specs
    def __init__(self, number_tiles):
        
        # Starting in the lowest point in the valley
        self.x = -pi/6
        self.velocity = 0
        
        # Set the tiles
        self.interval = np.linspace(0.0, 0.16, number_tiles+1)
        
        
    # Actually makes a move, and gets the environment into a new state
    def make_move(self, action):
        move = None
        if action[0] == 1: move = -1
        elif action[1] == 1: move = 0
        elif action[2] == 1: move = 1
        else: Exception("Action tuple does not make sense - " + action ) 
        
        self.velocity += 20*((0.001)*move - 0.0025*cos(3*self.x))
        self.x += self.velocity
        self.plot_state()
    
    
    # Hardcoded actions - at any time the car can move right, do nothing, or go left
    def get_actions(self):
        return (0,0,1), (0,1,0), (1,0,0)

    # The state is only the velocity of the mountain car, which is given as a list of tuples
    def get_state(self):
        state = []
        for ti in range(len(self.interval)-1):
            if self.interval[ti] <= self.velocity+0.08 <= self.interval[ti+1]:
                state.append(1)
            else:
                state.append(0)
        return tuple(state)

    # Plots the car on the graph
    def plot_state(self):

        x = np.arange(-2, np.pi ,0.1)   # start,stop,step
        y = np.cos(3*(x + pi / 2))
        
        plt.plot(x,y)
        plt.plot(self.x,cos(3*(self.x+pi/2)),'ro') 
        plt.show()
    
    # Returns true if the car reached the top
    def is_terminal(self):
        if self.x >= pi / 6 :
            return True
        return False
    
    # Returns positive reward for the terminal state, -1 for all other states
    def get_reward(self):
        if self.is_terminal():
            return 1000
        else:
            return -1


# Network
class NN():
    
    # Initialize the neural network
    def __init__(self, input_size):
        # Create the model
        self.model = tf.keras.Sequential()
        # Adding the input layer
        self.model.add(tf.keras.layers.InputLayer(input_shape = (input_size + 3, ) ))
        # Add the output layer with softmax
        self.model.add(tf.keras.layers.Dense( units = 1, activation='relu') )
        # Compile model        
        self.model.compile(loss = 'mean_squared_error')
        
    # Policy - returns the best action given the state
    def policy(self, state):

        # Generate SAPs given the current environment
        SAPs = [ ( state + a) for a in ((0,0,1), (0,1,0), (1,0,0)) ]  
        
        # Choose the best action
        return argmax(SAPs, self.evaluate)[-3:]

    # Evaluates the state action pair
    def evaluate(self, SAP):
        
        x = np.array(SAP)
        x = np.expand_dims(x,0)
        
        return self.model(x)        
        
    # Trains the network 
    def train(self, SAP, y):
        
        x = np.array(SAP)
        x = np.expand_dims(x,0)
        
        y = np.array(y)
        y = np.expand_dims(y,0)
        
        self.model.fit( x, y, epochs = 8, verbose = 0)
        


# SARSA

# Parameters
discount = 0.95
lrate = 0.1
number_tiles = 5
num_steps = 500
episodes = 100


# One run of a SARSA algorithm
def run(state,action, SAPs, env, step):
    
    # Add new action state
    if state + action not in SAPs: SAPs.append( state + action )
    
    # 1. Pick and do an action a, get reward r
    env.make_move(action)
    reward = env.get_reward()
    
    # 2. Choose the next action
    nextstate = env.get_state()
    nextaction = Q.policy(nextstate) if random() < 0.2 else choice(((0,0,1), (0,1,0), (1,0,0)))
    
    # 3. Calculate the change        
    delta = reward + discount*Q.evaluate(nextstate + nextaction) - Q.evaluate( state + action )
    
    # 4. Update every SAP
    for sap in SAPs:
        Q.train( sap, delta*lrate)
    
    # 5. If we are in the terminal state, we are done
    if (not env.is_terminal() ) and step > 1 :
        if env.is_terminal():
            print("yeah!")
        run(nextstate,nextaction, SAPs, env, step-1)
        

# Initialize the neural network, our Q(s,a) function
Q = NN(number_tiles)


# Atually run SARSA episodes number of times
for episode in range(episodes):
    
    print(episode)
    # visited SAPs are saved here
    SAPs = []

    # Initialize the state and pick an action
    env = Env(number_tiles)
    
    state = env.get_state()
    action = Q.policy(state)
    
    # Repeat until we have reached the final state or num_steps steps have been used up
    run(state, action, SAPs, env, num_steps)

    
    
    
    

