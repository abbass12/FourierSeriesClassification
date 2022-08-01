import numpy as np


ngrid=1500

idftmatdict={}
ngrid=1500
xx=np.linspace(-np.pi,np.pi,ngrid+1)
x=xx[0:-1]
train={}
test={}
SpectDict={'Trig':1,'Poly':2,'Exp':3}

N = [15,20,40,45,80,135,160,320,405,640,1215,1280,1500]


def calculateIdftMat():

    for n in N:
        nmodes = n
        idftmat = np.zeros((ngrid, nmodes), dtype=complex)
        nn = np.linspace(-nmodes / 2, nmodes / 2, nmodes + 1)
        for i in range(ngrid):
            for j in range(nmodes):
                idftmat[i][j] = np.e ** (1j * x[i] * nn[j])
        idftmatdict[n] = idftmat
        idftmat = np.zeros((ngrid, nmodes-1), dtype=complex)

        for i in range(ngrid):
            for j in range(nmodes-1):
                idftmat[i][j] = np.e ** (1j * x[i] * nn[j])
        idftmatdict[n-1] = idftmat



def main():
    calculateIdftMat()

if __name__!='__main__':
    calculateIdftMat()



def dft(N,M=40):
    #w=np.e**((2*np.pi*-1j)/N)
    power=np.zeros((M,N),dtype=complex)
    xj = np.linspace(-np.pi, np.pi, N)
    for i in range(M):
        for j in range(N):
            #power[i][j]=((-N/2)+i)*j
            power[i][j] = -1j*(i-M/2)*xj[j]
    #DFT=(w**power)/N
    DFT = np.e**power
    return DFT
algorithm={'ifft':1,'forloop':2,'precompute':3}
def FourierSeries(cn,X,method):
    N = len(cn)

    if algorithm[method]==3:
        return np.dot(idftmatdict[N], cn).real
    elif (algorithm[method]==2) or (N%2==0):
        fx = []
        for x in X:
            result = 0
            for i in range(int((-N / 2)), int((N / 2))):
                result = result + cn[int(i + (N / 2))] * (
                            np.e ** (1j * i * x))  # Calculate Fourier Series approximation using formula
            fx.append(result.real)
        return fx
    else:
        Cn=np.zeros(len(X),dtype=complex)
        Cn[0]=cn[int(N/2)]
        Cn[1:int(N/2)+1]=cn[int(N/2)+1:N+1]
        Cn[len(X)-int(N/2):len(X)]=cn[0:int(N/2)]
        fx=len(X)*np.fft.ifftshift(np.fft.ifft(Cn))
        return fx.real




def et(x,n,sign=1):
    if sign==1:
        return np.e**(1j*x*n)
    else:
        return np.e**(-1j*x*n)

def trigSig(n):

        Sipi=1.85193705198247
        sig=np.pi*np.sin((np.pi*np.abs(n))/(np.max(n)))/Sipi
        return sig
def polySig(n):
    sig=(np.pi*np.abs(n)/np.max(n))
    return sig
def expSig(n,N):
    alpha=2
    tau=np.linspace(1/N,1-(1/N),1000)
    res=tau[1]-tau[0]
    const = np.pi/( res*sum(np.e**(1/(alpha*tau*(tau-1)))) )
    sig = const*( np.abs(n)/np.max(n) )*np.e**(1/(alpha*(np.abs(n)/max(n))*((np.abs(n)/max(n))-1)));
    sig[n==0]=0
    sig[0]=0
    sig[-1]=0
    return sig

'''
    type determines which fourier coefficient to calculate
        trig:1
        exp:2
        poly:3
'''
def partialFourierSum(M2,M1,x,cn,type=1):
    type=SpectDict[type]
    n=np.linspace(-M1/2,M1/2,M1)
    D2=np.zeros((M2-1,M1),dtype=complex)
    for p in range(M2-1):
      for q in range(M1-1):
        D2[p][q]=(et(x[p],n[q]))
    if type==1:
        sig=trigSig(n)
    elif type==2:
        sig=polySig(n)
    elif type==3:
        sig=expSig(n,len(cn))
    SN=cn*(1j*np.sign(n)*sig)
    fx = np.dot(D2,SN)
    return fx.real

