import numpy as np
def addNoise(y,noiseParam,x):
    return y + (np.random.normal(0, noiseParam, len(x)))
def extractJump(x,y,a):
    fk=np.zeros(len(x))
    h=x[1]-x[0]
    jumpsx=x[np.abs(x+a)<=h/2][0],x[np.abs(x-a)<=h/2][0]
    lefty={}
    righty={}
    lefty[jumpsx[0]]=y[np.abs(x-(jumpsx[0]-h))<h/2][0]
    lefty[jumpsx[1]]=y[np.abs(x-(jumpsx[1]-h))<h/2][0]
    righty[jumpsx[0]]=y[np.abs(x-(jumpsx[0]+h))<h/2][0]
    righty[jumpsx[1]]=y[np.abs(x-(jumpsx[1]+h))<h/2][0]
    jL=righty[jumpsx[0]]
    jR=righty[jumpsx[1]]
    if(np.abs(lefty[jumpsx[0]])>np.abs(righty[jumpsx[0]])):
        jL=lefty[jumpsx[0]]*-1
    if(np.abs(lefty[jumpsx[1]])>np.abs(righty[jumpsx[1]])):
        jR=lefty[jumpsx[1]]*-1

    if(len(x[np.abs(x-a)<=h/2])==1):
        fk[np.abs(x+a)<=h/2]=jL
        fk[np.abs(x-a)<=h/2]=jR
    else:
        fk[np.abs(x+a)<=h/2][0]=jL
        fk[np.abs(x-a)<=h/2][0]=jR
    return fk