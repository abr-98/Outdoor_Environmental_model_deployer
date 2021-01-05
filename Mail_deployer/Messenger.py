#!/usr/bin/env python
# coding: utf-8

# In[1]:


import smtplib 


# In[2]:


from flask import Flask,request,Response


# In[3]:


app = Flask(__name__)


# In[4]:


@app.route("/")
def home():
    return "Hello from Home"


# In[5]:


@app.route('/send_email')
def send_email():
    email=request.args.get('email')
    body=request.args.get('body')
    smtp = smtplib.SMTP('smtp.gmail.com', 587) 
    smtp.starttls()
    smtp.login("cityprobe.report@gmail.com", "cityprobe123#")    
    sender="cityprobe.report@gmail.com"
    ids=email.split('_')
    for id_ in ids:
        message="From: cityprobe.report@gmail.com\n"
        subject="Subject: URGENT: Sensing mechanism disabled\n\n"
        send_ad="To: "+id_+"\n"
        message+=send_ad
        message+=subject
        message+=body 
        smtp.sendmail(sender, id_, message) 
    smtp.quit() 
    return Response(status = 200)


# In[6]:

#
#if __name__=='__main__':
#    app.run(host='0.0.0.0',port=5000)


# In[ ]:




