
from flask import Flask, request, render_template
import numpy as np
#import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
        timeelapsed=float(request.values["TimeElapsed"])
        pre_nullvotespercentage=float(request.values["pre.nullVotesPercentage"])
        invalidvotespercentage=float(request.values["invalidVotesPercentage"])
        validvotespercentage=float(request.values["validVotesPercentage"])
        voterspercentage=float(request.values["votersPercentage"])
        availablemandates= float(request.values["availableMandates"])
        numparishes = float(request.form["numParishes"])
        votes=float(request.values["Votes"])
        hondt=float(request.values["Hondt"])
        time=float(request.values["time"])
        territoryname= float(request.form["territoryName"])
        party = float(request.form["Party"])
        
        x=[[timeelapsed,pre_nullvotespercentage,invalidvotespercentage,validvotespercentage,voterspercentage,availablemandates,numparishes,votes,hondt,time,territoryname,party ]]
        
        output = model.predict(x)
        output=output.item()
        output=round(output)
        return render_template('result.html', prediction_text='Final Number of  Mandates are {}!'.format(output))

      
       

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9990)