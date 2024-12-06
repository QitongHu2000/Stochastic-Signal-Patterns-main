# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from sdeint import itoint
from scipy.integrate import solve_ivp
from scipy.io import mmread
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.3
eta=0.3
t_0=200#10 #1000
t_1=20
t0=0.005
n=200 #50000000 #1000000
alpha = 0.1
beta = 0.1
B=0.5
a=1.5
b=2

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def Part1(x):
    x=np.mat(x).T
    J_=-B*a*np.diag(np.power(x,a-1).T.tolist()[0])
    S_=alpha
    T_=b*np.diag((np.power(x,b-1)/np.power(np.power(x,b)+1,2)).T.tolist()[0])
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
        return np.mat(-B*np.power(x,a)+alpha*A*(np.power(x,b)/(1+np.power(x,b))))
    
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
        x[source]*=(1+gamma)
        sol=solve_ivp(Fun_1, [t_0,t_0+t_1], x, rtol=1e-13, atol=1e-13, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
        xs=sol.y.T
        ts=sol.t
        return np.mat(xs), np.array(ts)
        
    def time(xs,ts,eta):
        xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
        indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
        times=[]
        for i in range(len(indexs)):
            len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
            len_2=eta-xs[indexs[i],i]
            times.append(ts[indexs[i]]+len_2/len_1*(ts[indexs[i]+1]-ts[indexs[i]]))
        return np.mat(times)
    
    x,t=sim_first(A)
    xs,ts=sim_second(A,x.copy(),t,source)
    times=time(xs.copy(),ts.copy(),eta).tolist()[0]
    to_nodes=list(G_edge[source])
    times=[times[i]-t_0 for i in to_nodes]
    xn=xs[-1,:]
    return times, x, xn

def simulation_stochastic(A, source):
    def F(A,x):
        return np.mat(-B*np.power(x,a)+alpha*A*(np.power(x,b)/(1+np.power(x,b))))
    
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
        x_0=np.ones(np.shape(A)[0])*0.1
        sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-10, atol=1e-10) #odeint(Fun,x_0,t,args=(A,a,b))
        xs=sol.y.T
        x=xs[-1,:].tolist()
        return x
    
    def sim_second(A, x, source):
        x_now=x.copy()
        x_now[source]*=(1+gamma)
        x_now=np.array(x_now)
        t=np.linspace(0,t0,n)
        xs=itoint(Fun_1, Gun_1, x_now, t)
        return np.mat(xs)
        
    def time(xs,eta):
        xs=(xs-xs[0])/(xn-xs[0])
        indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
        times=[]
        # print(indexs)
        for i in range(len(indexs)):
            if(indexs[i]<n-1):
                # len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
                # len_2=eta-xs[indexs[i],i]
                times.append(indexs[i]) #+len_2/len_1)
            else:
                times.append(indexs[i])
        return np.mat(times)*t0/n
    
    x=sim_first(A)
    # print(x)
    xs=sim_second(A, x.copy(), source)
    times=time(xs.copy(),eta).tolist()[0]
    to_nodes=list(G_edge[source])
    times=[times[i] for i in to_nodes]
    return times, x

def J(x):
    return B*a*np.power(x,a-1)

def H2_(x):
    return b*np.power(x,b-1)/np.power(1+np.power(x,b),2)

def R_(x):
    return 1/alpha*B*a*np.power(x,a-1)

def G_(x):
    return beta# * (1-x)

def H_1(x):
    return alpha

def K(x):
    return G_(x)**2*R_(x)**2

def theory_continuous(x, source):
    to_nodes=list(G_edge[source])
    time_continuous=list()
    for node in to_nodes:
        time_continuous.append(1/(J(x[node])))
    return time_continuous

def theory_stochastic(x, source):
    to_nodes=list(G_edge[source])
    time_stochastic=list()
    for node in to_nodes:
        # print(H2_(x[source])/R_(x[node])*x[source]*gamma)
        # time_stochastic.append(-np.log(1-eta)/(B/(1-x[node])*(1+1/2*x[node]**2*beta**2/(alpha*(1-x[node])*gamma*x[source]))))
        time_stochastic.append(1/K(x[node]))
        # time_stochastic.append(-np.log(1-eta)/J(x[node])+1/4*G_(x[node])**2/J(x[node])**2/(H2_(x[source])/R_(x[node])*x[source]*gamma)**2)
        # print(G_(x[node])**2/J(x[node])**2/(H2_(x[source])/R_(x[node])*x[source]*gamma)**2)
    return time_stochastic

A_edge=mmread('../Networks/bio-celegans.mtx')
A_edge=np.mat(A_edge.toarray())
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
degrees=np.sum(A_edge,axis=1)

G_edge=nx.from_numpy_matrix(A_edge)

k=1
sources=[381,] #np.random.choice(G_edge.nodes, size=k)

degree=list()
times_continuous=list()
times_stochastic=list()
times_theory_continuous=list()
times_theory_stochastic=list()

ks=50
for source in sources:
    to_nodes=list(G_edge[source])
    print(source, len(to_nodes))
    for node in to_nodes:
        degree.append(degrees[node,0])
    time_continuous, x1, xn=simulation_continuous(A_edge, source)
    
    # times_continuous.extend(time_continuous)
    # times_stochastic_total=list()
    # for k in range(ks):
    #     print(k)
    #     time_stochastic, _=simulation_stochastic(A_edge, source)
    #     times_stochastic_total.append(time_stochastic)
    #     print(time_stochastic)
    # times_stochastic.extend(np.mean(times_stochastic_total,axis=0).tolist())
    
    time_theory_continuous=theory_continuous(x1, source)
    times_theory_continuous.extend(time_theory_continuous)
    
    time_theory_stochastic=theory_stochastic(x1, source)
    times_theory_stochastic.extend(time_theory_stochastic)
    
# save_dict(degree, 'R_celegans_degree'+str(int(100*a))+'_'+str(int(100*b)))
# save_dict(times_continuous, 'R_celegans_continuous'+str(int(100*a))+'_'+str(int(100*b)))
# save_dict(times_stochastic, 'R_celegans_stochastic'+str(int(100*a))+'_'+str(int(100*b)))
save_dict((np.mat(xn)-np.mat(x1))[0,to_nodes], 'R_celegans_Deltas_continuous'+str(int(100*a))+'_'+str(int(100*b)))
save_dict(times_theory_continuous, 'R_celegans_theory_continuous'+str(int(100*a))+'_'+str(int(100*b)))
save_dict(times_theory_stochastic, 'R_celegans_theory_stochastic'+str(int(100*a))+'_'+str(int(100*b)))