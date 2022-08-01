from FunctionDefinitions import *
from Fourier import *


SignalsDict={'Box':1,'Saw':2,'Exp':3,'Sin':4,'Gaus':5}

def Generate(signal,x,Amount,normalized=True,fourier=False,jump=False,N=40,type='Trig',Noise=False,noiseParameter=0.1,method='precompute',randomParam=True,a=0,b=0,c=0):
    signalOutput=[]
    length=Amount+2
    print(SignalsDict[signal])

    if(SignalsDict[signal]==1):
        if(randomParam):
            #a=np.linspace(np.pi/4,np.pi/2,len(x))
            a=np.linspace(.10,2.90,length)
            b=np.append(np.linspace(-100,-0.01,int(length/2)),np.linspace(0.01,100,int(length/2)))
            a=np.random.permutation(a)

            b=np.random.permutation(b)
        if fourier:
            for i in range(Amount):
                if(jump):
                    result=box(x,a[i],b[i],normalized=normalized,jump=True,type=type,fourier=True,N=N)
                    cn=result[0]
                    Jump=result[1]
                    signalOutput.append([FourierSeries(cn,x,method=method)[0:-1],Jump])
                else:
                    cn=box(x,a[i],b[i],normalized=normalized,fourier=True,N=N)
                    signalOutput.append(FourierSeries(cn,x,method=method))
                normalization=1#np.max(abs(F))
                #BoxSignal.append(F/normalization)
                #BoxSignalFourier.append(fx)

            return [signalOutput,a[0:Amount],b[0:Amount]]
        elif Noise:
            for i in range(Amount):
                F=box(x,a[i],b[i],noise=True,normalized=normalized,noiseParameter=noiseParameter)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount]]
        else:
            for i in range(Amount):
                F=box(x,a[i],b[i],jump=jump,normalized=normalized)
                #normalization=1#np.max(abs(F))
                #BoxSignal.append(F/normalization)
                #BoxSignalFourier.append(fx)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount]]
    elif(SignalsDict[signal]==2):
        if(randomParam):
            a=np.linspace(np.pi/4,np.pi/2,length)
            b=np.append(np.linspace(-100,-0.01,int(length/2)),np.linspace(0.01,100,int(length/2)))
            a=np.random.permutation(a)
            b=np.random.permutation(b)
        if fourier:
            for i in range(Amount):
                if(jump):
                    result=saw(x,a[i],b[i],normalized=normalized,jump=True,type=type,fourier=True,N=N)
                    cn=result[0]
                    Jump=result[1]
                    signalOutput.append([FourierSeries(cn,x,method=method)[0:-1],Jump])
                else:
                    cn=saw(x,a[i],b[i],normalized=normalized,fourier=True,N=N)
                    signalOutput.append(FourierSeries(cn,x,method=method))
                normalization=1#np.max(abs(F))
                #BoxSignal.append(F/normalization)
                #BoxSignalFourier.append(fx)

            return [signalOutput,a[0:Amount],b[0:Amount]]
        elif Noise:
            for i in range(Amount):
                F=saw(x,a[i],b[i],noise=True,normalized=normalized,noiseParameter=noiseParameter)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount]]
        else:
            for i in range(Amount):
                F=saw(x,a[i],b[i],jump=jump,normalized=normalized)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount]]
    elif(SignalsDict[signal]==3):
        if(randomParam):
            a=np.linspace(np.pi/4,np.pi/2,length)
            b=np.append(np.linspace(-1,-0.1,int(length/2)),np.linspace(0.1,1,int(length/2)))
            c=np.append(np.linspace(-3,-1.01,int(length/2)),np.linspace(-1.01,1,int(length/2)))
            a=np.random.permutation(a)
            b=np.random.permutation(b)
            c=np.random.permutation(c)
        if fourier:
            for i in range(Amount):
                if(jump):
                    result=exp(x,a[i],b[i],c[i],normalized=normalized,jump=True,type=type,fourier=True,N=N)
                    cn=result[0]
                    Jump=result[1]
                    signalOutput.append([FourierSeries(cn,x,method=method)[0:-1],Jump])
                else:
                    cn=exp(x,a[i],b[i],c[i],normalized=normalized,fourier=True,N=N)
                    signalOutput.append(FourierSeries(cn,x,method=method))
                normalization=1#np.max(abs(F))
                #BoxSignal.append(F/normalization)
                #BoxSignalFourier.append(fx)

            return [signalOutput,a[0:Amount],b[0:Amount],c[0:Amount]]
        elif Noise:
            for i in range(Amount):
                F=exp(x,a[i],b[i],c[i],noise=True,normalized=normalized,noiseParameter=noiseParameter)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount],c[0:Amount]]
        else:
            for i in range(Amount):
                F=exp(x,a[i],b[i],c[i],jump=jump,normalized=normalized)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount],c[0:Amount]]
    elif(SignalsDict[signal]==4):
        if(randomParam):
            a=np.linspace(np.pi/4,np.pi/2,length)
            b=np.append(np.linspace(-2*np.pi,-0.3,int(length/2)),np.linspace(0.3,2*np.pi,int(length/2)))
            c=np.append(np.linspace(-100,-0.1,int(length/2)),np.linspace(0.1,100,int(length/2)))
            a=np.random.permutation(a)
            b=np.random.permutation(b)
            c=np.random.permutation(c)
        if fourier:
            for i in range(Amount):
                if(jump):
                    result=sinu(x,a[i],b[i],c[i],normalized=normalized,jump=True,type=type,fourier=True,N=N)
                    cn=result[0]
                    Jump=result[1]
                    signalOutput.append([FourierSeries(cn,x,method=method)[0:-1],Jump])
                else:
                    cn=sinu(x,a[i],b[i],c[i],normalized=normalized,fourier=True,N=N)
                    signalOutput.append(FourierSeries(cn,x,method=method))
                normalization=1#np.max(abs(F))
                #BoxSignal.append(F/normalization)
                #BoxSignalFourier.append(fx)

            return [signalOutput,a[0:Amount],b[0:Amount],c[0:Amount]]
        elif Noise:
            for i in range(Amount):
                F=sinu(x,a[i],b[i],c[i],noise=True,normalized=normalized,noiseParameter=noiseParameter)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount],c[0:Amount]]
        else:
            for i in range(Amount):
                F=sinu(x,a[i],b[i],c[i],jump=jump,normalized=normalized)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount],c[0:Amount]]
    elif(SignalsDict[signal]==5):
        if(randomParam):
            a=np.linspace(np.pi/4,np.pi/2,length)
            #b=np.append(np.linspace(-10,-1,int(length/2)),np.linspace(1,10,int(length/2)))
            b=np.linspace(1,10,length)
            a=np.random.permutation(a)
            b=np.random.permutation(b)

        if fourier:
            for i in range(Amount):
                if(jump):
                    result=gaus(x,a[i],int(b[i]),normalized=normalized,jump=True,type=type,fourier=True,N=len(x),M=N)
                    cn=result[0]
                    Jump=result[1]
                    #signalOutput.append([FourierSeries(cn,x,method='ifft')[0:-1],Jump])
                    signalOutput.append([FourierSeries(cn, x, method=method)[0:-1], Jump])
                else:
                    cn=gaus(x,a[i],int(b[i]),normalized=normalized,fourier=True,N=len(x),M=N)
                    signalOutput.append(FourierSeries(cn,x,method=method))
                normalization=1#np.max(abs(F))
                #BoxSignal.append(F/normalization)
                #BoxSignalFourier.append(fx)

            return [signalOutput,a[0:Amount],b[0:Amount]]
        elif Noise:
            for i in range(Amount):
                F=gaus(x,a[i],int(b[i]),noise=True,normalized=normalized,noiseParameter=noiseParameter)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount]]
        else:
            for i in range(Amount):
                F=gaus(x,a[i],int(b[i]),jump=jump,normalized=normalized)
                signalOutput.append(F)
            return [signalOutput,a[0:Amount],b[0:Amount]]
#%%
