import unittest
import copy
from Ipython.display import Markdown, display
import numpy as numpy
from frozenlake import FrozenLakeEnv

def printmd(s):
    display(Markdown(s))

def policy_eval(env,policy,gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta=0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob,next_state,reward,done in env.P[s][a]:
                    Vs += action_prob*prob*(reward+gamma*V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

def q_from_v(env,V,s,gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state,reward,done in env.P[s][a]:
            q[a] += prob * (reward+gamma*V[next_state])
    return q

def policy_improvement(env,V,gamma=1):
    policy = np.zeros([env.nS,env.nA])/env.nA
    for s in range(env.nS):
        q = q_from_v(env,V,s,gamma)
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a],axis=0)/len(best_a)
    return policy

def policy_iteration(env,gamma=1,theta=1e-8):
    policy = np.ones([env.nS,env.nA])/env.nA
    while True:
        V = policy_evaluation(env,policy,gamma,theta)
        new_policy = policy_improvement(env,V)
        if(new_policy==policy).all():
            break;
        policy=copy.copy(new_policy)
    return policy,V

env = FrozenLake()
random_policy = np.ones([env.nS,env.nA])/env.nA

