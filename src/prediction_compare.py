import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO

API_URL_TEMPLATE = "https://data.norges-bank.no/api/data/EXR/B.{currency}.NOK.SP?format=csv&startPeriod={start_date}&endPeriod={end_date}&locale=en"

def download_exchange_rate_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        print("Data Found")
        return response.text
    else:
        print("Failed to download ecchange rate data from Norge Bank API.")
        return None

def preprocess_and_save_data(csv_data, output_file):
    df = pd.read_csv(StringIO(csv_data))
    #df['reportingBegin'] = pd.to_datetime(df['reportingBegin'])
    df.to_csv(output_file, index=False)
    print("Data Saved")

def corelation_based_predection(df, time_window):
    unique_dates = df['TIME_PERIOD'].unique()

    differences = []
    predictions = []
    correctness = []
        

    for i in range(len(unique_dates) - time_window):
        start_dt = unique_dates[i]
        end_dt = unique_dates[i + time_window]

        window_data = df[(df['TIME_PERIOD'] >= start_dt) & (df['TIME_PERIOD'])]

        start_value = window_data.iloc[0]['OBS_VALUE']
        end_value = window_data.iloc[-1]['OBS_VALUE']
        difference = end_value - start_value

        window_data['TIME_PERIOD'] = pd.to_datetime(window_data['TIME_PERIOD'])
        window_data['TIME_PERIOD'] = (window_data['TIME_PERIOD'] - window_data['TIME_PERIOD'].min()).dt.days

        correlation = window_data['TIME_PERIOD'].corr(window_data['OBS_VALUE'])

        if correlation > 0:
            prediction = 'Positive'
        else:
            prediction = 'Negative'

        correct_prediction = (np.sign(difference) == np.sign(correlation))
    
        correctness.append(correct_prediction)
            
    correct_percentage = (sum(correctness)/len(correctness))*100
    return correct_percentage


def mean_based_prediction(df,time_window):
    unique_dates = df['TIME_PERIOD'].unique()

    differences = []
    differences_file = []
    predictions = []
    correctness = []
        

    for i in range(len(unique_dates) - time_window):
        start_dt = unique_dates[i]
        end_dt = unique_dates[i + time_window]

        window_data = df[(df['TIME_PERIOD'] >= start_dt) & (df['TIME_PERIOD'])]

        mean_value = window_data['OBS_VALUE'].mean()
        if window_data.iloc[-1]['OBS_VALUE'] > mean_value:
            prediction = 1  # Predict a positive difference
        else:
            prediction = -1  # Predict a negative difference

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
    

    return correct_percentage


def threshold_based_prediction(df, time_window):
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

        start_value = window_data.iloc[0]['OBS_VALUE']
        end_value = window_data.iloc[-1]['OBS_VALUE']
        difference = end_value - start_value

    

        if i >= num_previous_windows:
            accumulated_difference -= differences[i - num_previous_windows]
        accumulated_difference += difference
            
        if i >= num_previous_windows:
            mean_difference = accumulated_difference / num_previous_windows
        else:
            # Handle the case when there are not enough previous windows
            mean_difference = 0
            
        threshold = threshold_fraction * mean_difference

        if difference > threshold:
            prediction = 1
        else:
            prediction = -1

        correct_prediction = (np.sign(difference) == np.sign(prediction))
        differences.append(difference)

            
        correctness.append(correct_prediction)
            
    
    correct_percentage = (sum(correctness)/len(correctness))*100

    return correct_percentage


def apply_rule(choice, df, window):
    if choice == 1:
        return corelation_based_predection(df, window)
    elif choice == 2:
        return mean_based_prediction(df, window)
    elif choice == 3:
        return threshold_based_prediction(df, window)
    else:
        print("Invalid selection")
        return None

def compare_rules_for_currencies(currencies, time_windows):
    comparison_results = []
    for currency in currencies:
        for window in time_windows:
            api_url = API_URL_TEMPLATE.format(currency=currency, start_date=start_date, end_date=end_date)
            csv_data = download_exchange_rate_data(api_url)

            if csv_data:
                output_file = f'../data/{currency}_data.csv'
                preprocess_and_save_data(csv_data, output_file)

                df = pd.read_csv(output_file, delimiter=';')

                for rule_choice in range(1, 4):  # Assuming you have 3 rules
                    accuracy = apply_rule(rule_choice, df, window)
                    comparison_results.append({
                        'Currency': currency,
                        'Time_Window': window,
                        'Rule': rule_choice,
                        'Accuracy': accuracy
                    })

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv('../data/comparison_prediction.csv', index = False)
    return comparison_df

if __name__ == "__main__":
    
    
    start_date = input("Enter the start date(YYYY-MM-DD): ")
    end_date = input("Enter the end date(YYYY-MM-DD): ")

    currencies_to_compare = ['USD', 'EUR', 'SEK', 'JPY']  # Add more currencies if needed
    time_windows_to_compare = [3, 7, 10]  # Add different time windows

    comparison_table = compare_rules_for_currencies(currencies_to_compare, time_windows_to_compare)
    print(comparison_table)
    grouped_data = comparison_table.groupby(['Currency', 'Time_Window', 'Rule']).mean().reset_index()

    # Plotting 
    
    for currency in currencies_to_compare:
        for rule_choice in range(1, 4):  # Assuming you have 3 rules
            rule_data = grouped_data[(grouped_data['Currency'] == currency) & (grouped_data['Rule'] == rule_choice)]
            plt.plot(rule_data['Time_Window'], rule_data['Accuracy'], label=f'Currency: {currency}, Rule: {rule_choice}')

    plt.xlabel('Time Window')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Prediction Rules for Different Currencies and Time Windows')
    plt.legend()
    plt.show() 
    
    