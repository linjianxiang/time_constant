from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA,MiniBatchSparsePCA 
from scipy import linalg
from sklearn import preprocessing
# from lag_select import
class Utility(object):
    
    @staticmethod
    def Augmentation(x,ta,h):

        if h == 0:
            x_shifted = x
            return x_shifted
        else:      
            N,m = np.shape(x)
            col = m*(h+1)
            Q = np.zeros((N-ta*h,col))
            for ii in range(h+1):
                Q[:,((ii)*m):(m*(ii+1))] = x[(ta*(h-ii)):(N-ta*(ii)),:]                            
            x_shifted = Q
        
        return x_shifted

    @staticmethod
    def AugmentReverse(Q,ta,h):
        if h == 0:
            x_dshifted = Q
            return x_dshifted
        else:      
            nrow,ncol = np.shape(Q)
            N=int(nrow+ta*h)
            M=int(ncol/(h+1))
            x_dshifted = np.zeros((N,M))
            x_sum = np.zeros((N,M))
            x_zero_add = np.zeros((N,ncol))
                        
           
            #last set of row in each augmented set
            # for ii in range(1,(h+2)):
            #     x_dshifted[(ta*(h-ii+1)):(N-ta*(ii-1)),:] = Q[:,((ii-1)*M):(M*ii)]           
            counter = h
            for i in range(1,(h+2)):
                # x_zero_add[counter:counter+nrow,(ta)*(i-1)*N:(ta)*i*N] = Q[:,(ta)*(i-1)*N:(ta)*i*N]
                x_zero_add[counter:counter+nrow,(ta)*(i-1)*M:(ta)*i*M] = Q[:,(ta)*(i-1)*M:(ta)*i*M]
                counter = counter - 1
            for i in range(h+1):
                x_sum = x_sum + x_zero_add[:,(ta)*(i)*M:(ta)*(i+1)*M]
            if h < N:
                for i in range(h):
                    x_dshifted[i:(i+1),:] = x_sum[i:(i+1),:]/(i+1)
                    x_dshifted[N-i-1:N-i,:] = x_sum[N-i-1:N-i,:]/(i+1)
                x_dshifted[h:N-h,:] = x_sum[h:N-h,:]/(h+1)
            else:
                print('h>=N')
        return x_dshifted


def DPCA_cal(x,h):      
    # x_normed = (x - x.min(0)) / x.ptp(0) #peak to peak normalization
    x_normed = preprocessing.scale(x) #Standardize a dataset along any axis #Center to the mean and component wise scale to unit variance
    x = x_normed
    x = Utility.Augmentation(x,1,h)
    pca = PCA()
    pca.fit(x)
    U, S, V = pca._fit(x)
    p_component_threshold = 0.9
    ratio_sum = 0
    for i in range(len(pca.explained_variance_ratio_)):
        if ratio_sum > p_component_threshold:
            break
        else:
            ratio_sum = ratio_sum + pca.explained_variance_ratio_[i]

    # p_component_num = sum(pca.explained_variance_ratio_>p_component_threshold)
    p_component_num = i
    p_hat = pca.components_[:,0:p_component_num]
    pi_hat = np.matmul(p_hat,np.transpose(p_hat))
    x_hat = np.matmul(x,pi_hat)

    p_til = pca.components_[:,p_component_num:]
    pi_til =  np.matmul(p_til,np.transpose(p_til))
    x_til = np.matmul(x,pi_til)

    x_reconstructed = x_hat + x_til 

    x_pc = np.matmul(x,p_hat)
    x_res = np.matmul(x,p_til)

    x_hat_reconst = Utility.AugmentReverse(x_hat,1,h)
    x_til_reconst = Utility.AugmentReverse(x_til,1,h)
    x_augment_reconst = x_hat_reconst+x_til_reconst

    plt.figure(1)
    plt.subplot(2,2, 1)
    plt.plot(x)
    plt.title('original x')
    plt.subplot(2,2, 2)
    plt.plot(x_hat)
    plt.title('x_hat')
    plt.subplot(2,2, 3)
    plt.plot(x_til)
    plt.title('x_til')
    plt.subplot(2,2, 4)
    plt.plot(x_reconstructed)
    plt.title('x_reconstructed')

    plt.figure(2)
    plt.subplot(2,2, 1)
    plt.plot(x)
    plt.title('original x')
    plt.subplot(2,2, 2)
    plt.plot(x_hat_reconst)
    plt.title('x_hat_reconst')
    plt.subplot(2,2, 3)
    plt.plot(x_til_reconst)
    plt.title('x_til_reconst')
    plt.subplot(2,2, 4)
    plt.plot(x_augment_reconst)
    plt.title('x_augment_reconst')

    plt.figure(3)
    plt.subplot(2,1,1)
    plt.plot(x_pc)
    plt.title('data in pricipal subspaces')
    plt.subplot(2,1,2)
    plt.plot(x_res)
    plt.title('data in residual subspaces')

    plt.show()
    return x_hat_reconst


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
            try:
                TH = np.linalg.pinv(RR).dot(FF)
                V[0,jj-1] = (v1 - FF.T.dot(TH))/Ncaps
                V[0,jj-1] = max(V[0,jj-1],eps)
                V[1:4,jj-1] = nnorig[j,:].T
                TH.shape = (dim_size,)
                theta[j,:] = TH
            except:
                print('pseudo inverse fail, svd didnt converge')
                V[0,jj-1] = V[0,jj-2]
                V[1:4,jj-1] = V[1:4,jj-2]

    
    #wrap up
    V = V.real
    #calculate delay 
    delay = np.argmin(V[0])
    return [delay,theta[delay,:]]
      


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
    

def arx_aic_test(x,na_max,nb_max,nk,num_output,num_input,data_size):
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
                        output_data = x[:,j] #output col 
                        input_data = x[:,num_output+i]  #input col
                        output_data = output_data - np.mean(output_data)
                        input_data = input_data - np.mean(input_data)
                        zIn = [output_data,input_data]
                        zIn = (np.array(zIn).T)
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
            # print(order_index)
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
            # print('the best order for esitmation output '+ str(j)+' by input' +str(i)+' is using na = '+str(na_order)+' nb= '+str(nb_order))
        print('na nb are',largest_na,largest_nb)
    return [largest_na,largest_nb]


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
    plt.figure(4)
    plt.subplot(num_input, num_output, nm+1)
    plt.plot(X,prediction)
    plt.plot(X,output[0:data_size+k])
    plt.title('output '+str(out_nm)+' by input '+str(in_nm))
    plt.legend(['prediction', 'real output'])
    plt.xlabel('n_th data')
    plt.ylabel('value')
    

def delay_arxstructd(x,na,nb,nk,num_data,num_output,plot_size):
    #reformat regression orders
    nkVec = (np.array(range(nk)))+1
    nkVec = np.array([nkVec]).T #convert nkvect to dim of nk*1
    naVec = na* np.ones([nk,1])
    nbVec = nb* np.ones([nk,1])
    nn = np.concatenate((naVec,nbVec,nkVec),axis=1)
    num_input = num_data-num_output

    delay = []
    theta = np.zeros([num_input*num_output,na+nb])
    #calculate delay 
    num = 0
    for i in range(num_input):
        for j in range(num_output):
            # output_data = df.iloc[0:len(df)][col_name[j]] #output col 
            # input_data = df.iloc[0:len(df)][col_name[num_output+i-1]]  #input col
            output_data =x[:,j] #output col 
            input_data = x[:,num_output+i]  #input col

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
            # print('To predict output' + str(j)+' by input '+str(i)+' the theta is '+str(theta[num]))
            num = num+1
        plt.figure(2)
        X = np.linspace(0,plot_size,plot_size)
        plt.plot(X,(input_data.T)[0:plot_size])

    delay_mean = np.mean(delay)
    delay_max = np.max(delay)
    delay_min = np.min(delay)
    print('ARX approach: the mean delay is'+' '+str(delay_mean))
    print('ARX approach: the max delay is'+' '+str(delay_max))
    print('ARX approach: the min delay is'+' '+str(delay_min))
    return delay




X = np.loadtxt(open('Simulation_Output_Model_Reactor2.csv','rb'),delimiter=',')
x = X[5:1000,0:5]
num_output = 3


# TE_simout = spio.loadmat('TE_simout.mat', squeeze_me=True)['simout']
# TE_xmv = spio.loadmat('TE_xmv.mat', squeeze_me=True)['xmv']
# outputd = TE_simout[:2000,[14,15,16,17,18]]
# inputd = TE_xmv[:2000,[7]]
# # inputd = TE_simout[:,[3]]
# x = np.append(outputd,inputd,axis =1)
#num_output = 5

#initial param
num_data = x.shape[1]
AIC_test_data_size =950#x.shape[0] 

na_max = 10
nb_max = 5
maxdelay = 40 #range of delay to search for the best result
plot_size = 950 #x.shape[0]

na = arx_aic_test(x,na_max,nb_max,1,num_output,num_data-num_output,AIC_test_data_size)[0]
print(na)
nb=na #Order of polynomial B(q) for ARX model

delay = delay_arxstructd(x,na,nb,maxdelay,num_data,num_output,plot_size)
delay = [i for i in delay if i < (0.9*maxdelay)] #remove unreliable values
delay = np.max(delay)
order = delay + na
print('choose the order to be '+str(order))    
DPCA_cal(x,order)

#fault simulation
# TE_simout_fault = spio.loadmat('TE_simout_fault.mat', squeeze_me=True)['simout']
# outputd_fault = TE_simout_fault[:,[14,15,16,17,18]]
# inputd_fault = TE_simout_fault[:,[3]]
# x_fault = np.append(outputd_fault,inputd_fault,axis =1)
# DPCA_cal(x_fault,order)