from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    try:
        nmax = max([max(na+np.ones([nm,1])),nbkm2])[0]
    except:
        nmax = max([max(na+np.ones([nm,1])),nbkm2])
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
    dim_size = int(nma+max(nb)[0])
    theta = np.zeros([nm,dim_size])
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
            TH.shape = (dim_size,)
            theta[j,:] = TH
    
    #wrap up
    V = V.real
    #calculate delay 
    delay = np.argmin(V[0])
    return [delay,theta[delay,:]]
      


def delay_arxstructd(filename,na,nb,nk,num_output,plot_size):
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
    theta = np.zeros([num_input*num_output,na+nb])
    #calculate delay 
    num = 0
    for i in range(num_input):
        for j in range(num_output):
            # output_data = df.iloc[0:len(df)][col_name[j]] #output col 
            # input_data = df.iloc[0:len(df)][col_name[num_output+i-1]]  #input col
            output_data = df.iloc[6000:9000][col_name[j]] #output col 
            input_data = df.iloc[6000:9000][col_name[num_output+i-1]]  #input col

            #mean max
            output_data = output_data/np.max(output_data)
            input_data = input_data/np.max(input_data)
            zIn = [output_data,input_data]
            zIn = np.array(zIn).T
            arx_return = arxstruc(zIn,zIn,nn)
            delay_temp = arx_return[0]
            theta[num] = arx_return[1]
            print('ARX approach: delay between'+' output'+str(j)+' and input'+str(i)+' is'+' '+str(delay_temp))
            delay.append(delay_temp)
            #plot ARX prediction
            ARX_plot(plot_size,output_data.T,input_data.T,theta[num],delay[num],na,nb,i,j,num,num_input,num_output)
            print('To predict output' + str(j)+' by input '+str(i)+' the theta is '+str(theta[num]))
            num = num+1
        plt.figure(2)
        X = np.linspace(0,plot_size,plot_size)
        plt.plot(X,(input_data.T)[0:plot_size])

    plt.show()
    delay_mean = np.mean(delay)
    delay_max = np.max(delay)
    delay_min = np.min(delay)
    print('ARX approach: the mean delay is'+' '+str(delay_mean))
    print('ARX approach: the max delay is'+' '+str(delay_max))
    print('ARX approach: the min delay is'+' '+str(delay_min))
    return delay


    
    

def ARX_plot(data_size,output,input,theta,delay,na,nb,in_nm,out_nm,nm,num_input,num_output):
    if na > (nb+delay):
        k = na
    else:
        k = (nb+delay)
    prediction = np.zeros([data_size+k,1])#predict from y[k] to y[data_size+k]
    for i in range(data_size):
        for j in range(na):
            prediction[i+k] = theta[j]*(-output[i+k-j-1])+prediction[i+k]  
        for q in range(na,na+nb):
            prediction[i+k] = prediction[i+k] +theta[q]*input[i+k-delay-q-na+1]
    X = np.linspace(0,data_size+k-1,data_size+k)
    plt.figure(1)
    plt.subplot(num_input, num_output, nm+1)
    plt.plot(X,prediction)
    plt.plot(X,output[0:data_size+k])
    plt.title('predicted output '+str(out_nm)+' by input '+str(in_nm))
    plt.legend(['prediction', 'real output'])
    plt.xlabel('n_th data')
    plt.ylabel('value')
    
def AIC_cal(output,input,theta,delay,na,nb,data_size):
    delay = 0
    if na > (nb+delay):
        k = na
    else:
        k = (nb+delay)
    prediction = np.zeros([data_size+k,1])#predict from y[k] to y[data_size+k]
    for i in range(data_size):
        for j in range(na):
            prediction[i+k] = theta[j]*(-output[i+k-j-1])+prediction[i+k]  
        for q in range(na,na+nb):
            prediction[i+k] = prediction[i+k] +theta[q]*input[i+k-delay-q-na+1]
    prediction.shape = (data_size+k,)
    prediction_error = prediction[0:data_size-1] - output[0:data_size-1]
    AIC = data_size*np.log(np.matmul(prediction_error,prediction_error.T)) + 2*(na+nb) + data_size*(1*np.log(2*np.pi)+1)
    AICc = AIC +2*(na+nb)*((na+nb)+1)/(data_size-(na+nb)-1)
    return [AIC,AICc]
    

def arx_aic_test(filename,na_max,nb_max,nk,num_output,num_input,data_size):
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


    #AIC init
    na_max = na_max+1
    nb_max = nb_max+1
    AIC = np.zeros([na_max*nb_max-1,num_input*num_output]) 
    AICc = np.zeros([na_max*nb_max-1,num_input*num_output]) 
    AIC_temp = np.zeros([num_input*num_output,])
    AICc_temp = np.zeros([num_input*num_output,])
    counter = 0
    for na in range(na_max):
        for nb in range(nb_max):
            if (na ==0 and nb ==0):
                na=na
            else:
                #format regression orders
                nkVec = (np.array(range(nk)))+1
                nkVec = np.array([nkVec]).T #convert nkvect to dim of nk*1
                naVec = na* np.ones([nk,1])
                nbVec = nb* np.ones([nk,1])
                nn = np.concatenate((naVec,nbVec,nkVec),axis=1)
                theta = np.zeros([num_input*num_output,na+nb])
                #calculate AIC
                num = 0
                for i in range(num_input):
                    for j in range(num_output):
                        # output_data = df.iloc[0:len(df)][col_name[j]] #output col 
                        # input_data = df.iloc[0:len(df)][col_name[num_output+i]]  #input col
                        output_data = df.iloc[6000:9000][col_name[j]] #output col 
                        input_data = df.iloc[6000:9000][col_name[num_output+i]]  #input col
                        output_data = output_data - np.mean(output_data)
                        input_data = input_data - np.mean(input_data)
                        zIn = [output_data,input_data]
                        zIn = np.array(zIn).T
                        arx_return = arxstruc(zIn,zIn,nn)
                        theta[num] = arx_return[1]
                        delay = 0
                        temp = AIC_cal(output_data.T,input_data.T,theta[num],delay,na,nb,data_size)
                        AIC_temp[num] = temp[0]
                        AICc_temp[num] = temp[1]
                        num = num+1
                AIC[counter,]=AIC_temp
                AICc[counter,]=AICc_temp
                counter = counter +1
    # print(AIC)
    largest_na = 0
    largest_nb = 0
    for i in range(num_input):
        for j in range(num_output):
            order_index = np.argmin(AIC[:,i]) +1
            print(order_index)
            nb_order = order_index%nb_max
            na_order = int(order_index/nb_max)
            if nb_max == 1:
                nb_order = nb_order
            elif order_index > nb_max:
                if nb_order == 0:
                    na_order = na_order -1
                    nb_order = nb_max
            else:
                na_order =0
                nb_order =order_index
            if na_order > largest_na:
                largest_na = na_order
            if nb_order > largest_nb:
                largest_nb = nb_order
            print('the best order for esitmation output '+ str(j)+' by input' +str(i)+' is using na = '+str(na_order)+' nb= '+str(nb_order))
    return [largest_na,largest_nb]


def lagh_select(filename,na,nb,maxdelay,num_output,plot_size):
    delay = delay_arxstructd(filename,na,nb,maxdelay,num_output,plot_size)
    for i in range(len(delay)):
        if delay[i] > 0.9*maxdelay:
            delay.remove(delay[i])
    delay = np.max(delay)
    order = delay + na
    print('choose the order to be '+str(order))        
    return order                     



#filename = 'testing_data.csv'
#filename = 'close_loop_data.csv'
filename = 'Simulation_Output_Model_ReactorO.csv'
num_data = 5
num_output = 3

na_max = 10
nb_max = 10
AIC_test_data_size = 500 #

#Order of polynomial A(q) for ARX model
na = arx_aic_test(filename,na_max,nb_max,1,num_output,num_data-num_output,AIC_test_data_size)[0]
nb=na #Order of polynomial B(q) for ARX model
# tuning parameters
maxdelay = 40 #range of delay to search for the best result
plot_size = 500
delay = lagh_select(filename,na,nb,maxdelay,num_output,plot_size)



#########input CSV file structure###########
# The data should be in following format
# Time  Output0 Ouput1 ... OutputN Input0 ...InputN