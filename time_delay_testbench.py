from __future__ import print_function
from estimation import estimate_parameters_4sid
import pandas as pd
import numpy as np
import statsmodels.api as sm
# import ssid
# reload(ssid)
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import scipy.signal as sgn
import matplotlib.pyplot as plt
import matplotlib as mpl
def get_input_output(name):
    df = pd.read_csv(name,index_col=None)
    # print(df.values)
    df.index = pd.to_datetime(df.index)
    # print(df.index)
    output_data = df.iloc[0:len(df)]['C_Aout'] #output col 
    input_data = df.iloc[0:len(df)]['F_in']  #input col
    return [output_data,input_data]


# def delay_AR_model(filename):
#     #import data
#     value = get_input_output(filename)
#     output_data = value[0]
#     input_data = value[1]

#     #apply ARX model 
#     model = ARX(output_data,input_data)

#     #calculate the delay 
#     delay = model.select_order(maxlag = 80, ic = 'aic', trend = 'c',method = 'cmle')
#     print('AR model delay is ',delay)

def delay_ARMA_model(filename,lag,delay):
    value = get_input_output(filename)
    output_data = value[0]
    input_data = value[1]

    best_aic = np.inf 
    best_order = None
    best_delay = None
    rng = range(lag)
    dl = range(delay)

    for d in dl:
        output_tmp = output_data[d:]
        input_tmp = input_data[:-(d)]
        for i in rng:
            for j in rng:
                try:
                    tmp_mdl = smt.ARMA(np.array(output_tmp), order=(i, j),exog =np.array(input_tmp) ).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, j)
                        best_delay = d
                except: continue


    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    print('delay',best_delay)

# def subspace_filter(y,u):
#     NumRows = 10
#     NumSig = 4
#     NumUCols = 2000
#     A,B,C,D,covID,S = ssid.N4SID(u,y,NumRows,NumUcols,NumSig)
#     print(A)
#     print(B)

def arxstruc(z,zv,nn):
    #z prameter
    [Ncaps,nz] = z.shape
    nu = nz-1
    nuv = nu
    #zv prameter
    [Ncapsv,nzv] = zv.shape
    [nm,nl] = nn.shape
    #Fix the orders for frequency domain data with the extra "initial" inputs
    nnorig = nn
    [nmorig,nlorig] = nnorig.shape
    #reassign na, nb, nk value
    na = np.array([nn[:,0]]).T
    nb = np.array([nn[:,1]]).T
    nk = np.array([nn[:,2]]).T
    #
    nma = max(na)[0]
    nbkm = (max(nb+nk)-nu)[0]
    nbord = nb
    nbkm2 = max(nbord)[0]
    nkm = min(nk)[0]
    n = int(nma + nbkm-nkm +nu)
    nmax = max([max(na+np.ones([nm,1])),nbkm2])[0]
    #
    R = np.zeros([n,n])
    F = np.zeros([n,1])

    Ncapv = Ncapsv
    nnm = nmax
    
    #generating matrix phi
    k = nnm
    jj = np.array(range(int(k),Ncapv+1))
    phi = np.zeros([jj.shape[0],n+1])
    #import output to the phi
    for kl in range(1,int(nma)+1):
        phi[:,kl-1] = -z[jj-kl-1,1-1]

    ss = nma
    for ku in range(1,max([nu,nuv])+1):
        nnkm = nkm
        nend = nbkm+nnkm
        
    for kl in range(int(nnkm),int(nend+1)):
        I = jj > kl
        phi[I,int(ss+kl-nnkm+1)-1] = z[jj[I]-kl-1,ku]
    
    ss = phi.shape[1]
    R = np.zeros([ss,ss])
    F = np.zeros([ss,1])
    v1 =1
    #generate matrix R and F
    R = R + np.matmul(phi.T,phi)
    F = F + phi.T.dot(np.array([z[jj-1,1-1]]).T)

    #compute loss function
    tmp_v1 = np.array([zv[int(nnm):Ncapv,1-1]])
    v1 = v1 + np.matmul(tmp_v1,tmp_v1.T)

    V = np.zeros([nlorig+1,nmorig])
    jj = 0
    eps = np.finfo(np.float).eps
    #computing cost function V
    for j in range(0,nm):
        estparno = na[j-1] + sum(nb[j])
        if estparno > 0: 
            jj = jj+1
            s = range(0,na[j])
            rs = nma
            ku = 0
            tmp_range = range(rs+nk[j]-nkm,rs + nb[j,ku]+nk[j]-nkm)
            s = s+tmp_range

            RR = R[s,:]
            RR = RR[:,s].real
            FF =F[s].real

            ##***********better solution can be used ************##
            TH = np.linalg.pinv(RR).dot(FF)
            V[0,jj-1] = (v1 - FF.T.dot(TH))/Ncaps
            V[0,jj-1] = max(V[0,jj-1],eps)
            V[1:4,jj-1] = nnorig[j,:].T
    
    #wrap up
    V = V.real
    #calculate delay 
    delay = np.argmin(V[0])
    print('arxstruc delay',delay)
    return delay
            



def delay_arxstructd(filename,na,nb,nk):
    #reformat regression orders
    nkVec = (np.array(range(nk)))+1
    nkVec = np.array([nkVec]).T #convert nkvect to dim of nk*1
    naVec = na* np.ones([nk,1])
    nbVec = nb* np.ones([nk,1])
    nn = np.concatenate((naVec,nbVec,nkVec),axis=1)
    #get data
    value = get_input_output(filename)
    zIn = np.array(value).T
    #calculate data
    arxstruc(zIn,zIn,nn)

def metstructd(filename,na,nb,nk):
    #reformat regression orders
    nkVec = (np.array(range(nk)))+1
    nkVec = np.array([nkVec]).T #convert nkvect to dim of nk*1
    naVec = na* np.ones([nk,1])
    nbVec = nb* np.ones([nk,1])
    nn = np.concatenate((naVec,nbVec,nkVec),axis=1)
    #get data
    value = get_input_output(filename)
    zIn = np.array(value).T
    output = zIn[:,0]
    input = zIn[:,1]
    # Output = np.array([zIn[:,0]]).T
    # Input = np.array([zIn[:,1]]).T
    # subspace_filter(Output,Input)
    a,c = estimate_parameters_4sid(zIn,10,10)
    # the c we got here has dim of 2
    c = (c/np.linalg.norm(c))
    c1 = c[0,:]
    c2 = c[1,:]
    filter_num = np.array([1,0,0,0,0,0,0,0,0,0])
    output_fti1 = sgn.lfilter_zi(filter_num,c1)
    output_ft1, _ = sgn.lfilter(filter_num,c1,output,zi=output_fti1*output[0])
    input_ft1, _ = sgn.lfilter(filter_num,c1,input,zi=output_fti1*output[0])
    #calculate data
    zIn1 = np.concatenate((np.array([output_ft1]),np.array([input_ft1])),axis = 0)
    arxstruc(zIn1.T,zIn1.T,nn)

    output_fti2 = sgn.lfilter_zi(filter_num,c2)
    output_ft2, _ = sgn.lfilter(filter_num,c2,output,zi=output_fti2*output[0])
    input_ft2, _ = sgn.lfilter(filter_num,c2,input,zi=output_fti2*output[0])
    zIn2 = np.concatenate((np.array([output_ft2]),np.array([input_ft2])),axis = 0)
    arxstruc(zIn2.T,zIn2.T,nn)




#filename = 'testing_data.csv'
#filename = 'close_loop_data.csv'
filename = 'Simulation_Output_Model_ReactorO.csv'
maxdelay = 80
delay_arxstructd(filename,10,5,maxdelay)
# delay_ARMA_model(filename,5,10)
# delay_AR_model(filename)
##metstructd(filename,10,5,80)