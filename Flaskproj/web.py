from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
   SL= float(request.values['SL'])
   SW= float(request.values['SW'])
   PL= float(request.values['PL'])
   PW= float(request.values['PW'])
   X=np.array([[SL,SW,PL,PW]])
   output=model.predict(X)
   output=output.item()
   if output==0:
        return render_template ('result.html',prediction_text="It is Iris-Satova")
   elif output==1:
        return render_template ('result.html',prediction_text="It is Iris-Versicolor")
   elif output==2:
        return render_template ('result.html',prediction_text="It is Iris-Virgica")
       
  
if __name__=='__main__':
    app.run(port=8000)
