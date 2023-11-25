import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from io import StringIO

API_URL_TEMPLATE = "https://data.norges-bank.no/api/data/EXR/B.{currency}.NOK.SP?format=csv&startPeriod={start_date}&endPeriod={end_date}&locale=en"

def download_exchange_rate_data(data):
    response = requests.get(api_url)
    if response.status_code == 200:
        print("Data Found")
        return response.text
    else:
        print("Failed to download ecchange rate data from Norge Bank API.")
        return None
    
def preprocess_and_save_data(data,output_file):
    df = pd.read_csv(StringIO(data))
    #df['reportingBegin'] = pd.to_datetime(df['reportingBegin'])
    df.to_csv(output_file, index=False)
    print("Data Saved")


if __name__ == "__main__":
    currency = input("Enter the specific currency code (i.g., USD): ")
    start_date = input("Enter the start date(YYYY-MM-DD): ")
    end_date = input("Enter the end date(YYYY-MM-DD): ")
    
    api_url = API_URL_TEMPLATE.format(currency=currency, start_date=start_date,end_date=end_date)
    print("Welcome to the currency prediction project")
    csv_data = download_exchange_rate_data(api_url)

    if csv_data:
        output_file = 'data/currency_data.csv'
        preprocess_and_save_data(csv_data,output_file) 
        

        #Time Window

        time_window = int(input("Enter the time window in days (e.g., 3) "))

        df = pd.read_csv(output_file,delimiter=';')

        #print(df['TIME_PERIOD'].unique())

        unique_dates = df['TIME_PERIOD'].unique()

        differences = []
        differences_file = []
        predictions = []
        correctness = []

        threshold_fraction = 0.8
        num_previous_windows = 5

        accumulated_difference = 0
        

        for i in range(len(unique_dates) - time_window):
            start_dt = unique_dates[i]
            end_dt = unique_dates[i + time_window]

            window_data = df[(df['TIME_PERIOD'] >= start_dt) & (df['TIME_PERIOD'])]

            mean_value = window_data['OBS_VALUE'].mean()

            if window_data.iloc[-1]['OBS_VALUE'] > mean_value:
                prediction = 1  # Predict a negative difference
            else:
                prediction = -1  # Predict a positive difference

            start_value = window_data.iloc[0]['OBS_VALUE']
            end_value = window_data.iloc[-1]['OBS_VALUE']
            difference = end_value - start_value

            
            correct_prediction = (np.sign(difference) == np.sign(prediction))

            #differences.append(difference)

            differences_file.append({
                                'Start_date': start_dt,
                                'End_date': end_dt,
                                'Difference': difference
                                })
            
            predictions.append({
                                'Start_date': start_dt,
                                'End_date': end_dt,
                                'Prediction': prediction
                                })
            
            correctness.append(correct_prediction)
            
        differences_file_df = pd.DataFrame(differences_file)
        predictions_df = pd.DataFrame(predictions)

        correct_percentage = (sum(correctness)/len(correctness))*100
        wrong_percentage = 100 - correct_percentage

            # Save the differences to a CSV file
        differences_file_df.to_csv('data/time_window_differences.csv', index=False)
        predictions_df.to_csv('data/time_window_prediction.csv', index = False)

        print("Time window differences saved to 'data/time_window_differences.csv'.")
        print("Time window predictions saved to 'data/time_window_predictions.csv'.")
        print(f"Percentage of correctly predicted time windows: {correct_percentage:.2f}%")

        #PieChart
        labels = ["Correctly Predict", "Wrongly Predict"]
        proportions = [correct_percentage, wrong_percentage]
        plt.pie(proportions, labels=labels, autopct="%1.1f%%")
        plt.title("Time window rate prediction")
        plt.show()

        

