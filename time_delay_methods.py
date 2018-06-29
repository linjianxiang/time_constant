from __future__ import print_function
import pandas as pd
import numpy as np

def get_input_output(name):
    df = pd.read_csv(name,index_col=None)
    # print(df.values)
    df.index = pd.to_datetime(df.index)
    # print(df.index)
    df.columns = ['Time','output1','output2','output3','input1','input2']
    output_data = df.iloc[0:len(df)]['output3'] #output col 
    input_data = df.iloc[0:len(df)]['input1']  #input col
    return [output_data,input_data]


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
    return delay
      


def delay_arxstructd(filename,na,nb,nk,num_output):
    #reformat regression orders
    nkVec = (np.array(range(nk)))+1
    nkVec = np.array([nkVec]).T #convert nkvect to dim of nk*1
    naVec = na* np.ones([nk,1])
    nbVec = nb* np.ones([nk,1])
    nn = np.concatenate((naVec,nbVec,nkVec),axis=1)
    #get data
    df = pd.read_csv(filename,index_col=None)
    df.index = pd.to_datetime(df.index)
    col_name = []
    num_input = 5-num_output
    i =0
    while i < num_output:
        col_name.append('output'+str(i))
        i = i+1
    i=0
    while i < num_input:
        col_name.append('input'+str(i))
        i = i+1 
    df.columns = ['Time',col_name[0],col_name[1],col_name[2],col_name[3],col_name[4]]
    delay = []
    #calculate delay 
    for i in range(num_input):
        for j in range(num_output):
            output_data = df.iloc[0:len(df)][col_name[j]] #output col 
            input_data = df.iloc[0:len(df)][col_name[num_output+i-1]]  #input col
            zIn = [output_data,input_data]
            zIn = np.array(zIn).T
            delay_temp = arxstruc(zIn,zIn,nn)
            print('ARX approach: delay between'+' output'+str(j)+' and input'+str(i)+' is'+' '+str(delay_temp))
            delay.append(delay_temp)
    delay_mean = np.mean(delay)
    delay_max = np.max(delay)
    delay_min = np.min(delay)
    print('ARX approach: the mean delay is'+' '+str(delay_mean))
    print('ARX approach: the max delay is'+' '+str(delay_max))
    print('ARX approach: the min delay is'+' '+str(delay_min))

def corr_delay_cal(input_data,output_data):
    data_length = input_data.shape[0]
    corrematrix = np.correlate(input_data,output_data,'full')
    corr_max = np.amax(corrematrix[data_length-1:]) #not including negtive corr
    corr_index = np.where(corrematrix == corr_max)[0][0]
    delay = corr_index - data_length + 1
    return delay

def delay_correlation(filename,num_output):
    df = pd.read_csv(filename,index_col=None)
    # print(df.values)
    df.index = pd.to_datetime(df.index)
    # print(df.index)
    col_name = []
    num_input = 5-num_output
    i =0
    while i < num_output:
        col_name.append('output'+str(i))
        i = i+1
    i=0
    while i < num_input:
        col_name.append('input'+str(i))
        i = i+1
     
    df.columns = ['Time',col_name[0],col_name[1],col_name[2],col_name[3],col_name[4]]
    delay = []
    for i in range(num_input):
        for j in range(num_output):
            output_data = df.iloc[0:len(df)][col_name[j]] #output col 
            input_data = df.iloc[0:len(df)][col_name[num_output+i-1]]  #input col
            delay_temp = corr_delay_cal(input_data,output_data)
            print('Correlation approach: delay between'+' output'+str(j)+' and input'+str(i)+' is'+' '+str(delay_temp))
            delay.append(delay_temp)
    delay_mean = np.mean(delay)
    delay_max = np.max(delay)
    delay_min = np.min(delay)
    print('Correlation approach: the mean delay is'+' '+str(delay_mean))
    print('Correlation approach: the max delay is'+' '+str(delay_max))
    print('Correlation approach: the min delay is'+' '+str(delay_min))

#filename = 'testing_data.csv'
#filename = 'close_loop_data.csv'
filename = 'Simulation_Output_Model_ReactorO.csv'
num_output = 3
maxdelay = 80
delay_arxstructd(filename,1,0,maxdelay,num_output)
delay_correlation(filename,num_output)



#input CSV file structure: 
# The data should be in following format
# Time  Output0 Ouput1 ... OutputN Input0 ...InputN