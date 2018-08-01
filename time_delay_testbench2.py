from __future__ import print_function
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.api import VAR, DynamicVAR
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA, IncrementalPCA,MiniBatchSparsePCA 
from scipy import linalg
from sklearn.decomposition import PCA
class Utility(object):
    
    @staticmethod
    def Augmentation(x,ta,h):

        if h == 0:
            x_shifted = x
            return x_shifted
        else:      
            N,m = np.shape(x)
            # col = m*h
            col = m*(h+1)
            Q = np.zeros((N-ta*h,col))
            # for ii in range(1,h+1):
            for ii in range(h+1):
                Q[:,((ii)*m):(m*(ii+1))] = x[(ta*(h-ii)):(N-ta*(ii)),:]                
                # Q[ta*(ii-1):,((ii-1)*m):(m*ii)] = x[(ta*(ii-1)):,:]                
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
                        
           
            #last set of row in each augmented set
            for ii in range( 1, (h+2)): 
                x_dshifted[(ta*(h-ii+1)):(N-ta*(ii-1)),:] = Q[:,((ii-1)*M):(M*ii)]           
                    
                
        return x_dshifted


    @staticmethod
    def MeanCentreData(x):
        #mean centre data
        mean = np.mean(x,axis=0)
        stdDev  = np.std(x,axis=0)
            
        x_meanCentred = (x-mean)/stdDev

        #for i in range(0,x.shape[0]):
            #x_meanCentred[i,:] = x[i,:] - mean
            #x_meanCentred[i,:] = np.divide(x_meanCentred[i,:],stdDev)

        return x_meanCentred


def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)
 
# shift &lt; 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift

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
            s = list(range(0,int(na[j])))
            rs = nma
            ku = 0
            tmp_range1 = list(range(int(rs+nk[j]-nkm),int(rs + nb[j,ku]+nk[j]-nkm)))
            s = s+tmp_range1

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
      

def autocorr(x,y, twosided=False, tapered=True):
    """
    Return (lags, ac), where ac is the estimated autocorrelation 
    function for x, at the full set of possible lags.
    
    If twosided is True, all lags will be included;
    otherwise (default), only non-negative lags will be included.

    If tapered is True (default), the low-MSE estimate, linearly
    tapered to zero for large lags, is returned.
    """
    nx = len(x)
    xdm = x - x.mean()
    ydm = y-y.mean()
    ac = np.correlate(xdm, ydm, mode='full')
    ac /= ac[nx - 1]
    lags = np.arange(-nx + 1, nx)
    if not tapered:  # undo the built-in taper
        taper = 1 - np.abs(lags) / float(nx)
        ac /= taper
    if twosided:
        return lags, ac
    else:
        return lags[nx-1:], ac[nx-1:]
    
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
    
    X = np.loadtxt('C:\\cloud\\Code\\Algo\\data\\Simulation_Output_Model_Reactor.csv',delimiter=',')
    w= np.zeros((5,5))
    
    for i in range(5):
        #for j in range(5):
        #lags, auto_x = autocorr(X[:7100,i],X[:7100,i])
         #w[i,j]=compute_shift(X[:7100,i],X[:7100,j])
        #plt.figure(i)
        #plt.plot(lags, auto_x, 'ro')
        #print(i,np.shape(lags),np.shape(auto_x))
    
    #Perform Dickey-Fuller test:
        #print ('Results of Dickey-Fuller Test:')
        dftest = adfuller(X[:2000,i], maxlag=100,autolag='AIC')
        dfoutput = pd.Series(dftest[0:3], index=['Test Statistic','p-value','#Lags Used'])
        print(dfoutput)                     
        
    
    #lag_acf = acf(ts_log_diff, nlags=20)
    #lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    '''modelv= DynamicVAR(X[:7100,:])    
    ftModel = modelv.fit
    print(ftModel)
    for i in range(5):
        # train autoregression
        compute_shift()
        model = AR(X[:7100,i])
        model_fit = model.fit(maxlag=205, disp=True)
        window = model_fit.k_ar
        coef = model_fit.params
        w[i]=model_fit.k_ar
        #print('window',window,i)
        #print('coef',coef,i)'''
    print('w',w)
    #calculate delay 
    for i in range(num_input):
        for j in range(num_output):
            output_data = df.iloc[0:len(df)][col_name[j]] #output col 
            input_data = df.iloc[0:len(df)][col_name[num_output+i-1]]  #input col
            print('Output',col_name[j])
            print('Input',col_name[num_output+i-1])
            zIn = [output_data,input_data]
            zIn = np.array(zIn).T
            delay_temp = arxstruc(zIn,zIn,nn)
            #new
            '''zIn = [input_data]
            zIn = np.array(zIn).T
            zOut= [output_data]
            zOut = np.array(zOut).T
            
            delay_temp = arxstruc(zIn,zOut,nn)'''
            
            
            print('ARX approach: delay between'+' output'+str(j)+' and input'+str(i)+' is'+' '+str(delay_temp))
            delay.append(delay_temp)
    
    print(delay)
    delay_mean = np.mean(delay)
    delay_max = np.max(delay)
    delay_min = np.min(delay)
    print('ARX approach: the mean delay is'+' '+str(delay_mean))
    print('ARX approach: the max delay is'+' '+str(delay_max))
    print('ARX approach: the min delay is'+' '+str(delay_min))

def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def EstimateLag(Xtr):
    lag=0
    Xmc= Utility.MeanCentreData(Xtr)
    prevS=0.001
    
    kSV = np.empty(60)
    kSVR = np.empty(60)
    pCom =np.empty(60)
    
    
    for index in range(0,61):
        
        Xtr= Utility.Augmentation(Xmc,1,lag)
        n,m = np.shape(Xtr)
    
        #U,S,V = linalg.svd(1/np.sqrt(n-1)*Xtr)	 # conducting svd on the training data
        U,S,V = linalg.svd(np.cov(Xtr))	 # conducting svd on the training data
        
        # print('m is ',m)
        if(index!=0):
            # pCom[lag] = (np.cumsum(S*100.0 / S.sum())).tolist().index([val for val in (np.cumsum(S*100.0 / S.sum())) if val>=98][0])
            # pCom[lag] =pCom[lag] /m    
            # print("pCom=",pCom[lag])
            # print("Shape=",(S[int(m-1)]),m,lag)
        
            kSV[lag]= (S[int(m-1)])              
            # if(prevS>0):
            #     kSVR[lag] = kSV[lag]/prevS
            #else:
             #   kSVR[lag] = -9999
            kSVR[lag] = kSV[lag]/prevS
            prevS = kSV[lag]
            lag=lag+1
        else:
            prevS = (S[int(m-1)])         
        
    
    #order ksv, ksvr        
    nkSV= normalize(kSV)
    nkSVR= normalize(kSVR)
    
    print(kSV)
    output=  np.sqrt(np.power(nkSV,2)+np.power(nkSVR,2))#,0.5)
    
    ordIndex =argsort(output)
    # print(np.argmin(output))
    # print(ordIndex)
    # plt.plot(pCom)
        
    for index in range(0,50,1):
        #goto min value
        if(ordIndex[index]>0):
            # print(ordIndex[index],kSVR[ordIndex[index]] ,kSVR[ordIndex[index]-1])
            # /if(kSVR[ordIndex[index]] <kSVR[ordIndex[index]-1]) and (kSVR[ordIndex[index]]!=9999):                
            if(kSVR[ordIndex[index]] <kSVR[ordIndex[index]-1]):   
                # print('the best l',ordIndex[index])             
                return ordIndex[index]
            
    return 0
    



X = np.loadtxt(open('Simulation_Output_Model_Reactor2.csv','rb'),delimiter=',')
h = 2
x = X[0:1500,0:5]
print(x.shape)
x = Utility.Augmentation(x,2,h)
pca = PCA()
pca.fit(x)
print(x.shape)
p_component_threshold = 0.95
ratio_sum = 0
print(pca.explained_variance_ratio_)
for i in range(len(pca.explained_variance_ratio_)):
    if ratio_sum > p_component_threshold:
        break
    else:
        ratio_sum = ratio_sum + pca.explained_variance_ratio_[i]

p_component_num = sum(pca.explained_variance_ratio_>p_component_threshold)
p_hat = pca.components_[:,0:p_component_num]
pi_hat = np.matmul(p_hat,np.transpose(p_hat))
x_hat = np.matmul(x,pi_hat)

p_til = pca.components_[:,p_component_num:]
pi_til =  np.matmul(p_til,np.transpose(p_til))
x_til = np.matmul(x,pi_til)

x_reconstructed = x_hat + x_til 

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
plt.show()

