#!/usr/bin/env python
# coding: utf-8

### Caller API returns 3 paths: Deterministic, probabilistic 30, probabilistic 50
# In[1]:

## importing libraries
import googlemaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import PIL


# In[2]:

## Durgapur's location
START_LONG=87.269568
START_LAT=23.534924
END_LONG=87.321653
END_LAT=23.565774
ACTIONS=8


# In[3]:

## Creating Divisions 
#### Dividing the whole city in NxN divisions

#### Currently N=16 

def create_divisions(N):
    Y_Axis=np.linspace(START_LAT,END_LAT,N)
    X_Axis=np.linspace(START_LONG,END_LONG,N)
    Longitudes,Latitudes=np.meshgrid(X_Axis,Y_Axis)
    df_lat_long=pd.DataFrame(list(zip(Latitudes.flatten(),Longitudes.flatten())),columns=['Latitudes','Longitudes'])
    return df_lat_long


# In[4]:
### Calculating Distance metrics

### Euclidean distance calculated

def euclidean_distance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5


# In[5]:
### Passing pollution_prediction
### Passing N-no.of divisions: Help to provide granularity control
### Passing Destination

def add_reward_pollution_column(pollution_prediction,N,destination):
    df_lat_long=create_divisions(N)
    
    GOAL_STATE=np.argmin(df_lat_long[['Latitudes','Longitudes']].apply(lambda tup: euclidean_distance(tup[0],tup[1],destination[0],destination[1]),axis=1).values)
    #### Goal state defined
    df_lat_long['Pol_rewards']=-1*(np.array((PIL.Image.fromarray(pollution_prediction)).resize((N,N),PIL.Image.NEAREST)).flatten())
    #### For ubuntu: If Error Use this:
    #df_lat_long['Pol_rewards']=-1*(np.array((PIL.Image.fromarray(pollution_prediction)).astype(np.uint8).fromarray(arr)).resize((N,N),PIL.Image.NEAREST)).flatten())
    ### Defined 
    df_lat_long.loc[GOAL_STATE,'Pol_rewards']=10
    ## Defined goal state rewards
    return df_lat_long


# In[6]:

### combines the distance based rewards and the pollution based rewards

MAX_AQI=3
MIN_AQI=1
def return_combined_data(pollution_prediction,N,destination,alpha):
    matrix=[]
    df_state_details=add_reward_pollution_column(pollution_prediction,N,destination)
    with open('Recorded_details.json', 'r') as openfile: 
        fixed_data = json.load(openfile)
    for i in range(N):
        for j in range(N):
            point_det=[]
            data=fixed_data.get(str(N*i+j))
            point_det.append(data['Id'])
            for action in range(ACTIONS):
                if data['directions'][str(action)]['distance']!=0:
                    data['directions'][str(action)]['cost']=alpha*(data['directions'][str(action)]['cost'])+(1-alpha)*((df_state_details.iloc[N*i+j]['Pol_rewards']-MIN_AQI)/(MAX_AQI-MIN_AQI))
            else:
                data['directions'][str(action)]['cost']=df_state_details.iloc[N*i+j]['Pol_rewards']
            
            point_det.append(data['directions'])
            matrix.append(point_det)
    df_matrix=pd.DataFrame(matrix,columns=['Id','directions'])
    df_state_details=pd.concat([df_matrix,df_state_details],axis=1)
    
    return df_state_details
        


# In[7]:


#### ALGORITHM
def policy_iteration(P,R,STATES):
    Q=np.zeros((ACTIONS,STATES,1))
    Q_bk=np.random.random((ACTIONS,STATES,1))
    V=np.zeros((STATES,1))
    gamma=0.999
    while(np.abs(Q-Q_bk).sum()>0.000001):
        Q_bk=Q.copy()
        Q=R+gamma*np.matmul(P,V)
        policy=np.argmax(Q,axis=0)
        V=np.max(Q,axis=0)
    return policy


# In[8]:


### ALGORITHM EXECUTER
def return_policy(pollution_prediction,destination,N,alpha):
    df_state_details=return_combined_data(pollution_prediction,N,destination,alpha)
    STATES=len(df_state_details)
    P=np.zeros((ACTIONS,STATES,STATES))
    R=np.zeros((ACTIONS,STATES,1))
    for state in range(STATES):
        mapping=df_state_details.iloc[state]['directions']
        for action in range(ACTIONS):
            move=mapping.get(str(action))
            P[action,state,move['id']]=1
            R[action,state]=move['cost']
    policy=policy_iteration(P,R,STATES)
    return policy,df_state_details 


# In[9]:


def get_single_source_path_deterministic(source,destination,pollution_prediction,alpha=0.3,N=16): 
    policy,df_state=return_policy(pollution_prediction,destination,N,alpha)    
    SOURCE_STATE=np.argmin(df_state[['Latitudes','Longitudes']].apply(lambda tup: euclidean_distance(tup[0],tup[1],source[0],source[1]),axis=1).values)
    GOAL_STATE=np.argmin(df_state[['Latitudes','Longitudes']].apply(lambda tup: euclidean_distance(tup[0],tup[1],destination[0],destination[1]),axis=1).values)
    Path=[]
    AQI_SUM=0
    state=SOURCE_STATE
    while state!=GOAL_STATE:
        Tuples=df_state.iloc[state]['directions'][str(policy[state][0])]['Tuples_list']
        AQI_SUM+=df_state.iloc[state]['Pol_rewards']
        for tup in Tuples:
            Path.append(tup)
        state=df_state.iloc[state]['directions'][str(policy[state][0])]['id']
    
    map_to_return={}
    map_to_return['Path']=list(Path)
    map_to_return['AQI_sum']=int(AQI_SUM)
    return map_to_return
    


# In[10]:


def get_single_source_path_probabilistic_30(source,destination,pollution_prediction,alpha=0.3,N=16): 
    policy,df_state=return_policy(pollution_prediction,destination,N,alpha)    
    SOURCE_STATE=np.argmin(df_state[['Latitudes','Longitudes']].apply(lambda tup: euclidean_distance(tup[0],tup[1],source[0],source[1]),axis=1).values)
    #dest_lat,dest_long=get_lat_long(destination)
    GOAL_STATE=np.argmin(df_state[['Latitudes','Longitudes']].apply(lambda tup: euclidean_distance(tup[0],tup[1],destination[0],destination[1]),axis=1).values)
    Path=[]
    state=SOURCE_STATE
    AQI_SUM=0
    while state!=GOAL_STATE:
        if np.random.random()<=0.7:
            action=policy[state][0]
        else:
            action=np.random.randint(0,8)
            
        Tuples=df_state.iloc[state]['directions'][str(action)]['Tuples_list']
        AQI_SUM+=df_state.iloc[state]['Pol_rewards']
        for tup in Tuples:
            Path.append(tup)
        state=df_state.iloc[state]['directions'][str(action)]['id']
    
    map_to_return={}
    map_to_return['Path']=list(Path)
    map_to_return['AQI_sum']=int(AQI_SUM)
    return map_to_return
    


# In[11]:


def get_single_source_path_probabilistic_50(source,destination,pollution_prediction,alpha=0.3,N=16): 
    policy,df_state=return_policy(pollution_prediction,destination,N,alpha)    
    SOURCE_STATE=np.argmin(df_state[['Latitudes','Longitudes']].apply(lambda tup: euclidean_distance(tup[0],tup[1],source[0],source[1]),axis=1).values)
    #dest_lat,dest_long=get_lat_long(destination)
    GOAL_STATE=np.argmin(df_state[['Latitudes','Longitudes']].apply(lambda tup: euclidean_distance(tup[0],tup[1],destination[0],destination[1]),axis=1).values)
    Path=[]
    state=SOURCE_STATE
    AQI_SUM=0
    while state!=GOAL_STATE:
        if np.random.random()<=0.5:
            action=policy[state][0]
        else:
            action=np.random.randint(0,8)
            
        Tuples=df_state.iloc[state]['directions'][str(action)]['Tuples_list']
        AQI_SUM+=df_state.iloc[state]['Pol_rewards']
        for tup in Tuples:
            Path.append(tup)
        state=df_state.iloc[state]['directions'][str(action)]['id']
    
    map_to_return={}
    map_to_return['Path']=list(Path)
    map_to_return['AQI_sum']=int(AQI_SUM)
    return map_to_return
    


# In[12]:


def all_results(source,destination,pollution_prediction,alpha=0.3):
    source_l=source.split('_')
    source=(float(source_l[0]),float(source_l[1]))
    dest_l=destination.split('_')
    destination=(float(dest_l[0]),float(dest_l[1]))
    
    deterministic_path_details=get_single_source_path_deterministic(source,destination,pollution_prediction,alpha)
    probabilistic_30_path_details=get_single_source_path_probabilistic_30(source,destination,pollution_prediction,alpha)
    probabilistic_50_path_details=get_single_source_path_probabilistic_50(source,destination,pollution_prediction,alpha)
    
    mapping_routes={}
    mapping_routes['deterministic']=deterministic_path_details
    mapping_routes['probabilistic_30']=probabilistic_30_path_details
    mapping_routes['probabilistic_50']=probabilistic_50_path_details
    
    return mapping_routes
    


# In[13]:


from flask import Flask,request,Response,jsonify


# In[14]:


app = Flask(__name__)


# In[15]:


@app.route('/')
def home():
    return "Hello from home"


# In[16]:


@app.route('/get_route')
def get_route():
    source=request.args.get('src')
    destination=request.args.get('dst')
    alpha=float(request.args.get('weight'))
    arr=np.array([
    [1,2,1,3],
    [2,1,1,3],
    [1,1,1,2],
    [2,2,3,3]
    ],dtype=np.uint8)
    paths=all_results(source,destination,arr,alpha)
    return jsonify(paths)
    


# In[17]:


app.run()


# In[ ]:


