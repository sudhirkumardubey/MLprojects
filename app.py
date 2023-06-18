from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Application name
application = Flask(__name__) # Entry point

app = application

## Route for our home page

@app.route('/')
def index():
    return render_template('index.html') # It will search for 'templates' folder

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET' :
        return render_template('home.html') # simple model fill that we need to provide to our model to get prediction
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
    
        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline =PredictPipeline()
        predict_pipeline.predict(pred_df)

        results = predict_pipeline.predict(pred_df)

        # return to our home page
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
