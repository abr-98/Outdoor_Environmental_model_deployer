#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Crawler Meteoblue


# In[1]:


def crawler_meteoblue():
    from selenium import webdriver
    driver = webdriver.Chrome('chromedriver_linux64/chromedriver')
    driver.get("\")

    driver.find_element_by_xpath("/html/body/div[2]/div/form/div/input").click() #accept and continue

    button=driver.find_element_by_class_name("bloo")

    ######Click on checkbox#########

    print("B")

    l=['//*[@id="params"]/optgroup[2]/option[3]',
       '//*[@id="params"]/optgroup[6]/option[1]',
       '//*[@id="params"]/optgroup[4]/option[1]',
       '//*[@id="params"]/optgroup[4]/option[2]',
       '//*[@id="params"]/optgroup[3]/option[2]',
       '//*[@id="params"]/optgroup[3]/option[4]',
       '//*[@id="params"]/optgroup[3]/option[3]']

    for e in l:
        driver.find_element_by_xpath(e).click()
        print("C")
    print("A")
    driver.find_element_by_xpath('//*[@id="wrapper-main"]/div/main/div/div[2]/form/div[4]/div[3]/div/input').click()


# In[2]:


### Getting Data Meteoblue


# In[3]:


def get_data_meteoblue():
    import os
    import shutil
    fldr1='/home/abhijit/Downloads/'
    files=os.listdir(fldr1)
    to_reach='Meteoblue_data/'
    file_name='env_vars'
    for file in files:
        #print(file)
        if 'dataexport' in file:
            print(file)
            fle=file
    #state="mv "+fldr1+fle+" "+to_reach+file_name
    #print(state)
    shutil.move(fldr1+fle,to_reach)


# In[4]:


### Preprocessing Meteoblue


# In[5]:


def pre_process_meteoblue():
    import pandas as pd
    import numpy as np
    import os
    files=os.listdir('Meteoblue_data/')
    for file in files:
        fle=file
    df=pd.read_excel('Meteoblue_data/'+fle)
    a=df.iloc[8]
    arr=np.array(df.iloc[9:,:])
    a=np.array(df.iloc[8,:])

    l=[]
    l.append(a[0])
    for i in a:
        if 'timestamp' in i:
            continue
        else:
            l.append(i[9:])
    final=pd.DataFrame(arr,columns=l)
    required=['timestamp','Wind Speed [10 m]','Wind Direction [10 m]','Wind Gust','Cloud Cover High [high cld lay]','Mean Sea Level Pressure [MSL]']
    req=final[required]
    cols=['timestamp','Wind Speed  [10 m above gnd]','Wind Direction  [10 m above gnd]','Wind Gust  [sfc]','High Cloud Cover  [high cld lay]','Mean Sea Level Pressure  [MSL]']
    req_a=np.array(req)
    final_n=pd.DataFrame(req_a,columns=cols)
    final_n.to_csv('temp_meteoblue.csv',index=False)


# In[6]:


### Crawler NIT IOT


# In[7]:


def crawler_IOT():
    import pandas as pd
    import os
    from selenium import webdriver
    driver = webdriver.Chrome('chromedriver_linux64/chromedriver')
    driver.get("")
    button=driver.find_element_by_class_name("col-md-12")
    butt=button.find_element_by_class_name("text-center")
    links = driver.find_elements_by_link_text('Save')
    for link in links:
        link.click()


# In[8]:


### Getting Data IOT


# In[9]:


def get_data_IOT():
    import os
    import shutil
    fldr1='/home/abhijit/Downloads/'
    files=os.listdir(fldr1)
    to_reach='IOT_data/'
    fle=[]
    for file in files:
        #print(file)
        if 'Device' in file:
            print(file)
            fle.append(file)
    #state="mv "+fldr1+fle+" "+to_reach+file_name
    #print(state)
    for f in fle:
        shutil.move(fldr1+f,to_reach)


# In[10]:


#### Preprocessing IOT


# In[11]:


def preprocess_IOT(date_in='NA'):
    print(date_in)
    import os
    import pandas as pd
    fldr='IOT_data'
    files=os.listdir(fldr)
    devices_data={}
    devices=['Device-1.xls','Device-2.xls','Device-3.xls','Device-4.xls']
    for file in devices:
        ac=fldr+'/'+file
        print(ac)
        df=pd.read_html(ac)
        date=[]
        df_ac=df[0]
        i=0
        while i<len(df_ac):
            date.append(df_ac.iloc[i]['Date'].split(':')[0])
            i+=1
        df_ac['Date_req']=date
        if 'NA' in date_in:
            d_temp=df_ac.iloc[0]['Date_req']
            #print("A")
        else:
            d_temp=date_in
            #print("B")
        df_temp=df_ac[df_ac['Date_req']==d_temp]
        time_stamp=d_temp
        reqs=['Temperature','Humidity']
        df_t=df_temp[reqs]
        means=df_t.mean(axis=0)

        number=(file.split('.')[0]).split('-')[1]
        devices_data[number]={}
        devices_data[number]['Time']=time_stamp
        devices_data[number]['Temperature']=means['Temperature']
        devices_data[number]['Humidity']=means['Humidity']

    data_df=pd.DataFrame(devices_data) 
    df_f=data_df.transpose()
    print(df_f)
    df_f=df_f.reset_index()      
    df1=pd.read_csv("temp_meteoblue.csv")        
    i=0
    date_req=[]
    while i<len(df1):
        date_req.append((df1.iloc[i]['timestamp']).split(':')[0])
        i+=1
    df1['Time']=date_req
    df_com=df1[df1['Time']==d_temp]
    result = df_f.merge(df_com, on='Time',how='left')
    result=result.drop(['Time'],axis=1)

    result.to_csv('Final.csv',index=False)


# In[12]:


### Flush memories


# In[13]:


def flush_memo():
    import os
    fldrs=['IOT_data/','Meteoblue_data/']
    
    for fldr in fldrs:
        files=os.listdir(fldr)
        for file in files:
            print(file)
            os.remove(fldr+'/'+file)


# In[14]:


### Caller


# In[15]:


import os
import time


# In[21]:


def caller(date_in='2020-12-19 11'):
    flush_memo()

    flag=True
    while flag:
        try:
            crawler_meteoblue()
            flag=False
        except:
            pass

    time.sleep(2)
    get_data_meteoblue()

    pre_process_meteoblue()

    crawler_IOT()
    print("Downloading")
    while sum([os.path.exists(f'/home/abhijit/Downloads/Device-{e}.xls') for e in range(1,8)])<7:
        pass
    print("finished")

    get_data_IOT()

    preprocess_IOT(date_in)


# In[22]:



# In[ ]:




