# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from sdeint import itoint
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.01
eta=0.3
t_0=10#10 #1000
t_1=1
t0=1
n=500 #50000000 #1000000
alpha = 0.1
beta = 0.1
B=0.25

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def Part1(x):
    x=np.mat(x).T
    J_=np.diag((-B/(1-x)).T.tolist()[0])
    S_=np.diag((1-x).T.tolist()[0])
    T_=alpha
    return np.array(J_+S_*A_edge*T_)
    # return np.array(np.mat(-B*x+alpha*np.multiply(1-x,A_edge*x)).T.tolist()[0])
        
def Part2(x):
    x=np.mat(x).T
    # print(np.shape(x)[0])
    # print(np.identity(np.shape(x)[0]))
    return beta * np.identity(np.shape(x)[0])
    # return beta * np.array(np.diag((1-x).T.tolist()[0]))

def Part3(x):
    x=np.mat(x).T
    return np.zeros(shape=(np.shape(x)[0],np.shape(x)[0])) #np.array(beta * np.diag(x.T.tolist()[0]))

def simulation_continuous(A,source):
    def F(A,x):
        return np.mat(-B*x+alpha*np.multiply(1-x,A*x))
    
    def Fun(t,x):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(t,x,source):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[source]=0
        return dx
    
    def sim_first(A):
        x_0=np.ones(np.shape(A)[0])*0.1
        sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-10, atol=1e-10) #odeint(Fun,x_0,t,args=(A,a,b))
        xs=sol.y.T
        t=sol.t
        x=xs[-1,:].tolist()
        return x, t[-1]
    
    def sim_second(A,x,t_0,source):
        x[source]+=gamma
        sol=solve_ivp(Fun_1, [t_0,t_0+t_1], x, rtol=1e-10, atol=1e-10, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
        xs=sol.y.T
        ts=sol.t
        return np.mat(xs), np.array(ts)
    
    x,t=sim_first(A)
    xs,ts=sim_second(A,x.copy(),t,source)
    return xs,ts

def simulation_stochastic(A, source):
    def F(A,x):
        return np.mat(-B*x+alpha*np.multiply(1-x,A*x))
    
    def Fun(t,x):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(x,t):
        F_1=Part1(x)
        F_1[source,0].fill(0)
        return F_1.dot(x)
    
    def Gun_1(x,t):
        G=Part2(x)
        G[source][source]=0
        return np.array(G)
    
    def sim_first(A):
        x_0=np.zeros(np.shape(A)[0])*1e-3
        # sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-13, atol=1e-13) #odeint(Fun,x_0,t,args=(A,a,b))
        # xs=sol.y.T
        # x=xs[-1,:].tolist()
        return x_0
    
    def sim_second(A, x, source):
        x_now=x.copy()
        x_now[source]+=gamma
        x_now=np.array(x_now)
        ts=np.linspace(0,t0,n)
        xs=itoint(Fun_1, Gun_1, x_now, ts)
        return np.mat(xs), np.mat(ts)
    
    x=sim_first(A)
    xs,ts =sim_second(A, x.copy(), source)
    return xs,ts

A_edge=np.mat(loadmat('../Networks/ER1.mat')['A'])
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
degrees=np.sum(A_edge,axis=1)

G_edge=nx.from_numpy_matrix(A_edge)

k=1
sources=[126,] #np.random.choice(G_edge.nodes, size=k)

degree=list()
xs_continuous=list()
ts_continuous=list()
xs_stochastic=list()
ts_stochastic=list()

ks=10
for source in sources:
    to_nodes=list(G_edge[source])
    print(source, len(to_nodes))
    for node in to_nodes:
        degree.append(degrees[node,0])
    xs1,ts1=simulation_continuous(A_edge, source)
    xs_continuous.extend(np.mean(xs1,axis=1).T.tolist()[0])
    ts_continuous.extend(ts1.tolist())
    ts_continuous=[i-t_0 for i in ts_continuous]
    
    xs_stochastic_total=list()
    for k in range(ks):
        print(k)
        xs1_stochastic,ts1_stochastic=simulation_stochastic(A_edge, source)
        xs_stochastic_total.append((xs1_stochastic*degrees/np.mean(degrees)).T.tolist()[0])
    xs_stochastic.extend(np.mean(xs_stochastic_total,axis=0).tolist())
    ts_stochastic.extend(ts1_stochastic.tolist())
    
save_dict(xs_continuous, 'E_ER_xs_continuous'+str(int(100*B)))
save_dict(ts_continuous, 'E_ER_ts_continuous'+str(int(100*B)))
save_dict(xs_stochastic, 'E_ER_xs_stochastic'+str(int(100*B)))
save_dict(ts_stochastic, 'E_ER_ts_stochastic'+str(int(100*B)))