#!/usr/bin/env python
# coding: utf-8

### Storing the regularized path distances from API and storing the intermediate (lat,long) tuples between two states 
# In[1]:

## importing libraries
import googlemaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import PIL
import json


# In[2]:


#gmaps = googlemaps.Client(key='')


# In[3]:

## Durgapur's location
START_LONG=87.269568
START_LAT=23.534924
END_LONG=87.321653
END_LAT=23.565774


# In[4]:

## Creating Divisions 
#### Dividing the whole city in NxN divisions

#### Currently N=16 
def create_divisions(N):
    Y_Axis=np.linspace(START_LAT,END_LAT,N)
    X_Axis=np.linspace(START_LONG,END_LONG,N)
    Longitudes,Latitudes=np.meshgrid(X_Axis,Y_Axis)
    df_lat_long=pd.DataFrame(list(zip(Latitudes.flatten(),Longitudes.flatten())),columns=['Latitudes','Longitudes'])
    return df_lat_long


# In[5]:
### Calculating Distance metrics

### Euclidean distance calculated
def euclidean_distance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

### Calculated distance from maps   ####Currently deactivated
def transport_distance(lat1,long1,lat2,long2,mode="driving"):
    directions_result = gmaps.directions((lat1,long1),(lat2,long2),mode=mode)
    return float(directions_result[0]['legs'][0]['distance']['text'].split()[0])


# In[6]:

### Dummy: returns tuples list
### In defn: POINTS (X1,Y1) and (X2,Y2)
### CHECK THE PARAMETER PASSING

INTERMIDIATE_TUPLES=5
def return_tuples(x1,y1,x2,y2):
    list_to_return=[]
    axis_Y=np.linspace(x1,x2,INTERMIDIATE_TUPLES)    ##Latitudes
    axis_X=np.linspace(y1,y2,INTERMIDIATE_TUPLES)    ##Longitudes
    for index in range(len(axis_Y)):
        list_to_return.append((axis_Y[index],axis_X[index]))
    list_to_return.reverse()
    
    return list_to_return


# In[7]:
### POINTS (X1,Y1) and (X2,Y2)
### Dummy Tuple Creater and distance

def euc_dist_plus_tuples(x1,y1,x2,y2):

    dist=euclidean_distance(x1,y1,x2,y2)
    tup_list=return_tuples(x1,y1,x2,y2)
    
    return dist,tup_list


# In[8]:


ACTIONS=8


# In[9]:


### Providing movements according to actions
## 0-> i, j+1 (right)
## 1-> i+1,j+1 (diagonal upper)
## 2-> i+1,j (up)
## 3-> i+1,j-1 (left up diagonal)
## 4-> i,j-1 (left)
## 5-> i-1,j-1 (left right diagonal)
## 6-> i-1,j (down)
## 7-> i-1,j+1 (down,right)
def get_move(action,position):
    if action==0:
        return (position[0],position[1]+1)
    if action==1:
        return (position[0]+1,position[1]+1)
    if action==2:
        return (position[0]+1,position[1])
    if action==3:
        return (position[0]+1,position[1]-1)
    if action==4:
        return (position[0],position[1]-1)
    if action==5:
        return (position[0]-1,position[1]-1)
    if action==6:
        return (position[0]-1,position[1])
    if action==7:
        return (position[0]-1,position[1]+1)


# In[10]:
### Initializing all points details 
### Df_matrix columns= ['Index_Lat','Index_Long','Id','directions']
### Df_df_state_details=['Latitudes','Longitudes']


def init_all_details(N):
    df_lat_long=create_divisions(N)
    matrix=[]  
    index=0
    for i in range(0,N):
        for j in range(0,N):
            point_det=[i]
            point_det.append(j)
            point_det.append(N*i+j)
            map_directions={}
            for action in range(ACTIONS):
                map_directions[action]={}
                map_directions[action]['id']=N*i+j
                map_directions[action]['cord']=get_move(action,(i,j))
                map_directions[action]['distance']=0
                map_directions[action]['Tuples_list']=[]
                map_directions[action]['cost']=0
            point_det.append(map_directions)
            index+=1
            matrix.append(point_det)
    df_matrix=pd.DataFrame(matrix,columns=['Index_Lat','Index_Long','Id','directions'])
    df_state_details=pd.concat([df_matrix,df_lat_long],axis=1)
    
    return df_state_details


# In[17]:

### Regularizing the distance matrix using min-max regularization
### Storing the data in a file 
### format:json
"""
 state:{
    Latitude
    longitude
    coordinate_x
    coordinate_y
    ID
    directions{
        action 0 {
            tuples list
            next id
            coordinates of next point
        }
        action 1{
            .................
        }
        .......

    }
}

"""
def get_all_details(N=16):
    df_state_details=init_all_details(N)
    index=0
    travel_cost=[]
    record={}
    for i in range(N):
        for j in range(N):
            mapping=df_state_details.iloc[index]['directions']
            for action in range(ACTIONS):
                x_cord,y_cord=((df_state_details.iloc[index]['directions'])[action])['cord']
                if x_cord<0 or y_cord<0 or x_cord>N-1 or y_cord>N-1:
                    travel_cost.append(0)
                    continue
                
                target_index=df_state_details[(df_state_details['Index_Lat']==x_cord) & (df_state_details['Index_Long']==y_cord)].iloc[0]['Id']
                mapping[action]['id']=int(target_index)
                distance,tuples=euc_dist_plus_tuples(df_state_details.iloc[target_index]['Latitudes'],df_state_details.iloc[target_index]['Longitudes'],df_state_details.iloc[index]['Latitudes'],df_state_details.iloc[index]['Longitudes'])
                mapping[action]['Tuples_list']=list(tuples)
                travel_cost.append(distance)
            df_state_details.iloc[index]['directions']=mapping
            index+=1
    travel_cost=np.array(travel_cost)
    index=0
    for i in range(N):
        for j in range(N):
            mapping=df_state_details.iloc[N*i+j]['directions']
            for action in range(ACTIONS):
                if travel_cost[index]!=0:
                    mapping[action]['distance']=float(travel_cost[index])
                    mapping[action]['cost']=float(-(travel_cost[index]-np.min(travel_cost))/(np.max(travel_cost)-np.min(travel_cost)))
                index+=1
            df_state_details.iloc[N*i+j]['directions']=mapping
            record[N*i+j]={}
            record[N*i+j]['i']=i
            record[N*i+j]['j']=j
            record[N*i+j]['Id']=N*i+j
            record[N*i+j]['Latitude']=df_state_details.iloc[N*i+j]['Latitudes']
            record[N*i+j]['Longitude']=df_state_details.iloc[N*i+j]['Longitudes']
            record[N*i+j]['directions']=dict(mapping)
    #print(record)
    with open("Recorded_details.json", "w") as outfile: 
        json.dump(record, outfile)        
                          


# In[18]:


get_all_details()


# In[ ]:




