from flask import Flask, request, render_template
import os
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import logging

application = Flask(__name__)
app = application

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collecting data from the form
            data = CustomData(
                Cycle_Index=float(request.form.get('Cycle_Index')),
                Discharge_Time_s=float(request.form.get('Discharge_Time_s')), 
                Decrement_3_6_3_4V_s=float(request.form.get('Decrement_3_6_3_4V_s')), 
                Max_Voltage_Dischar_V=float(request.form.get('Max_Voltage_Dischar_V')), 
                Min_Voltage_Charg_V=float(request.form.get('Min_Voltage_Charg_V')), 
                Time_at_4_15V_s=float(request.form.get('Time_at_4_15V_s')),
                Time_constant_current_s=float(request.form.get('Time_constant_current_s')), 
                Charging_time_s=float(request.form.get('Charging_time_s'))
            )

            # Converting data to DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.debug(f"Data received from form: {pred_df}")

            # Predicting using the pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            logging.debug(f"Prediction results: {results}")

            return render_template('home.html', results=results[0])
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            return render_template('home.html', results="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
