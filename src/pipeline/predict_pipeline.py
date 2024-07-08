import pandas as pd
import os
import sys
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = load_object(file_path=model_path)
        self.preprocessor = load_object(file_path=preprocessor_path)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 Cycle_Index: float,
                 Discharge_Time_s: float,
                 Decrement_3_6_3_4V_s: float,
                 Max_Voltage_Dischar_V: float,
                 Min_Voltage_Charg_V: float,
                 Time_at_4_15V_s: float,
                 Time_constant_current_s: float,
                 Charging_time_s: float):
        self.Cycle_Index = Cycle_Index
        self.Discharge_Time_s = Discharge_Time_s
        self.Decrement_3_6_3_4V_s = Decrement_3_6_3_4V_s
        self.Max_Voltage_Dischar_V = Max_Voltage_Dischar_V
        self.Min_Voltage_Charg_V = Min_Voltage_Charg_V
        self.Time_at_4_15V_s = Time_at_4_15V_s
        self.Time_constant_current_s = Time_constant_current_s
        self.Charging_time_s = Charging_time_s

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Cycle_Index": [self.Cycle_Index],
                "Discharge Time (s)": [self.Discharge_Time_s],
                "Decrement 3.6-3.4V (s)": [self.Decrement_3_6_3_4V_s],
                "Max. Voltage Dischar. (V)": [self.Max_Voltage_Dischar_V],
                "Min. Voltage Charg. (V)": [self.Min_Voltage_Charg_V],
                "Time at 4.15V (s)": [self.Time_at_4_15V_s],
                "Time constant current (s)": [self.Time_constant_current_s],
                "Charging time (s)": [self.Charging_time_s]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
