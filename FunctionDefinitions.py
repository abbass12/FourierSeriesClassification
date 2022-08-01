import numpy as np
from Fourier import *
from FunctionOperatons import *
def box(x,a=2,b=5,normalized=True,jump=False,type='Trig',noise=False,noiseParameter=0.1,fourier=False,N=40):
    if fourier:

        k=np.linspace(int((-N/2)),int((N/2)),N+1)

        fkhat=np.zeros(len(k))
        fkhat[k==0]=(2*a*b)
        fkhat[k!=0]=((2*b*np.sin(a*k[k!=0]))/k[k!=0])
        y=fkhat/(2*np.pi)

        if normalized:
            if jump:
                y=y[0:-1]/np.max(np.abs(box(x,a,b,normalized=False)))

                return y,partialFourierSum(1500,len(y),x,y,type)
            return y[0:-1]/np.max(np.abs(box(x,a,b,normalized=False)))

        return y
    y=np.zeros(len(x))
    y[np.abs(x)<a]=b
    if noise:
        if normalized:
            y=y/np.max(abs(y))
        return addNoise(y,noiseParameter,x)
    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk=extractJump(x,y,a)
            return y[0:-1],fk[0:-1]
        return y/np.max(abs(y))
    return y
def saw(x,a=2,b=5,normalized=True,jump=False,type='Trig',noise=False,noiseParameter=0.1,fourier=False,N=40):
    if fourier:
        k=np.linspace(int((-N/2)),int((N/2)),N+1)
        fkhat=np.zeros(len(k)).astype(complex)
        fkhat[k==0]=0
        fkhat[k!=0]=(2*1j*b*(np.sin(a*k[k!=0])-a*k[k!=0]*np.cos(a*k[k!=0])))/(k[k!=0]**2)
        y=fkhat/(2*np.pi)

        if normalized:
            if jump:
                y=y[0:-1]/np.max(np.abs(saw(x,a,b,normalized=False)))

                return y,partialFourierSum(1500,len(y),x,y,type)
            return y[0:-1]/np.max(np.abs(saw(x,a,b,normalized=False)))

        return y
    y=np.zeros(len(x))
    y[np.abs(x)<a]=-b*x[np.abs(x)<a]
    if noise:
        if normalized:
            y=y/np.max(abs(y))
        return addNoise(y,noiseParameter,x)
    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1],fk[0:-1]

        return y/np.max(abs(y))
    return y
def exp(x,a=2,b=2,c=-1,normalized=True,jump=False,type='Trig',noise=False,noiseParameter=0.1,fourier=False,N=40):
    if fourier:
        k=np.linspace(int((-N/2)),int((N/2)),N+1).astype(complex)
        fkhat=np.zeros(len(k)).astype(complex)
        fkhat[k==0]=(2*((a*b*c)+np.sinh(a*b)))/b
        fkhat[k!=0]=((2*c*np.sin(a*k[k!=0]))/k[k!=0])+((2*np.sinh(a*(b+(1j*k[k!=0]))))/(b+(1j*k[k!=0])))
        y=fkhat/(2*np.pi)

        if normalized:
            if jump:
                y=y[0:-1]/np.max(np.abs(exp(x,a,b,c,normalized=False)))

                return y,partialFourierSum(1500,len(y),x,y,type)
            return y[0:-1]/np.max(np.abs(exp(x,a,b,c,normalized=False)))

        return y
    y=np.zeros(len(x))
    y[np.abs(x)<a]=c+np.e**(-b*x[np.abs(x)<a])
    if noise:
        if normalized:
            y=y/np.max(abs(y))
        return addNoise(y,noiseParameter,x)
    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1],fk[0:-1]
        return y/np.max(abs(y))
    return y
def sinu(x,a=2,b=2,c=-1,normalized=True,jump=False,type='Trig',noise=False,noiseParameter=0.1,fourier=False,N=40):
    if fourier:
        k=np.linspace(int((-N/2)),int((N/2)),N+1).astype(complex)
        fkhat=np.zeros(len(k)).astype(complex)
        fkhat[k==b]=((1j*c*np.sin(2*a*b))/(2*b))-(1j*a*c)
        fkhat[k==-b]=(1/2)*1j*c*((2*a)-((np.sin(2*a*b))/b))
        fkhat[k!=b]=(2*1j*c*((b*np.cos(a*b)*np.sin(a*k[k!=b]))-(k[k!=b]*np.sin(a*b)*np.cos(a*k[k!=b]))))/((b**2)-(k[k!=b]**2))
        y=fkhat/(2*np.pi)

        if normalized:
            if jump:
                y=y[0:-1]/np.max(np.abs(sinu(x,a,b,c,normalized=False)))

                return y,partialFourierSum(1500,len(y),x,y,type)
            return y[0:-1]/np.max(np.abs(sinu(x,a,b,c,normalized=False)))

        return y
    y=np.zeros(len(x))
    y[np.abs(x)<a]=c*np.sin(b*x[np.abs(x)<a])
    if noise:
        if normalized:
            y=y/np.max(abs(y))
        return addNoise(y,noiseParameter,x)
    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1],fk[0:-1]
        return y/np.max(abs(y))
    return y
def gaus(x,a=2,b=2,normalized=True,jump=False,type='Trig',noise=False,noiseParameter=0.1,fourier=False,N=40,M=40):
    if fourier:
        y=((np.dot(dft(N,M),gaus(x,a,b)))/N)
        if normalized:
            if jump:
                y=y[0:-1]/np.max(np.abs(gaus(x,a,b,normalized=False)))

                return y[0:-1],partialFourierSum(1500,len(y),x,y,type)
            return y[0:-1]/np.max(np.abs(gaus(x,a,b,normalized=False)))

        return y
    y=np.zeros(len(x))
    y[np.abs(x)<a]=np.e**(-a*(x[np.abs(x)<a]**(2*b)))
    y=np.nan_to_num(y)
    if noise:
        if normalized:
            y=y/np.max(abs(y))
        return addNoise(y,noiseParameter,x)
    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1],fk[0:-1]
        return y/np.max(abs(y))
    return y
