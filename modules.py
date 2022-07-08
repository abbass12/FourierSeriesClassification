import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
#np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def plot(fig,x,y,name):
    fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name=name))

