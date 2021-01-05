from flask import Flask,jsonify
app=Flask(__name__)

@app.route("/")
def home():
    return "hello from edited home"

@app.route("/get_data")
def get_data():
    dic={'a':123,'b':345}
    return jsonify(dic)

@app.route("/hi/<name>")
def hi(name):
    return "hi "+name+" bye"


#app.run(host='0.0.0.0',port=3000)