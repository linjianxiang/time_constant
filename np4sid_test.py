import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.signal as sp
import numpy.random as rnd
import ssid
reload(ssid)
# Just a helper for defining plants
def generalizedPlant(A,B,C,D,Cov,dt):
    CovChol = la.cholesky(Cov,lower=True)
    NumStates = len(A)

    B1 = CovChol[:NumStates,:]
    B2 = B

    Bbig = np.hstack((B1,B2))

    D1 = CovChol[NumStates:,:]
    D2 = D
    Dbig = np.hstack((D1,D2))

    P = (A,Bbig,C,Dbig,dt)
    return P

# Define the plant. Here it is a coupled spring-mass-damper system with forces and position measurements.
dt = .1
k1 = 1.
k2 = 2.
c1 = 2.
c2 = 2.
m1 = 1.
m2 = 1.
Kmat = np.array([[k1+k2,-k2],
                 [-k2,k2]])
Cmat = np.diag([c1,c2])

Mmat  = np.diag([m1,m2])


Ac = np.zeros((4,4))

Ac[:2,2:] = np.eye(2)
Ac[2:,:2] = -la.solve(Mmat,Kmat,sym_pos=True)
Ac[2:,2:] = -la.solve(Mmat,Cmat,sym_pos=True)
A = np.eye(4) + dt * Ac

Bc = np.zeros((4,2))
Bc[2:] = la.inv(Mmat)


B = dt * Bc
C = np.zeros((2,4))
C[:,:2] = np.eye(2)

NumInputs = B.shape[1]
NumOutputs = C.shape[0]

D = np.zeros((NumOutputs,NumInputs))

NumStates = len(A)
NumOutputs,NumInputs = D.shape

Q = .1 * np.eye(NumStates) * dt 
S = np.zeros((NumStates,NumOutputs))
R = .1 * np.eye(NumOutputs) / dt

CovTop = np.hstack((Q,S))
CovBot = np.hstack((S.T,R))
Cov = np.vstack((CovTop,CovBot))

P = generalizedPlant(A,B,C,D,Cov,dt)

# Generate an input to the plant.
# Here we are using two independent Ornstein-Uhlenbeck processes.
theta = .001
AU = (1- dt * theta) * np.eye(NumInputs)
BU = 10 * np.eye(NumInputs) * np.sqrt(dt)
CU = np.eye(NumInputs)
DU = np.zeros((NumInputs,NumInputs)) / np.sqrt(dt)
# Sampling rate of 1
USys = (AU,BU,CU,DU,dt)
NumURows = 10
NumUCols = 2000
NumU = 2 * NumURows + NumUCols - 1
Time = np.arange(NumU)
uu = rnd.randn(NumU,NumInputs)
# u is the actual input
tu,u,xu = sp.dlsim(USys,uu)

w = rnd.randn(NumU,NumStates+NumOutputs)
bigU = np.hstack((w,u))
tout,y,xout = sp.dlsim(P,bigU)
plt.plot(tout,u)
plt.plot(tout,y)

# Identify matrices using the Standard N4SID algorithm
AID,BID,CID,DID,CovID,S = ssid.N4SID(u.T,y.T,NumURows,NumUCols,4)
# For conistency with the Subspace ID literature, 
# we transpose the signals so that each input and output instance is a column vector
# This is opposite from the lsim convention

# Check the singular values
plt.plot(S / S.sum())

# Check how the computed impulse resonse compares.
plt.figure()
NhSteps = 200
tout,hTrue = sp.dimpulse((A,B,C,D,dt),t=dt * np.arange(NhSteps))
tout,hID = sp.dimpulse((AID,BID,CID,DID,dt),t=dt*np.arange(NhSteps))
for i in range(NumOutputs):
    for j in range(NumInputs):
        plt.figure()
        plt.plot(tout,hTrue[j][:,i])
        plt.plot(tout,hID[j][:,i])