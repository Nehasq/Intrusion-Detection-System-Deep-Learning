from flask import Flask, request, render_template_string
from flask import jsonify
import numpy as np
import pandas as pd
#import tensorflow as tf
from keras.models import load_model
from joblib import load
from flask import render_template
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__,static_folder='assets')

@app.route('/')
def index():
    return render_template('index.html')

# Load the preprocessor and model
preprocessor = load('preprocessor.joblib')  
model = load_model('autoencoder.h5') 
# Load the pre-trained LSTM model from the H5 file
lstm_model = load_model('intrusion_detection_LSTM_model_6.h5') 


def preprocess_data(file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
   
    #X = df.drop('label', axis=1)

    # Apply the preprocessor loaded from joblib
    X_preprocessed = preprocessor.transform(df)
    
    #y = df['label']
    #y_binary = (y != 'normal').astype(int)
    return df, X_preprocessed


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.csv'):

        df,X_preprocessed = preprocess_data(file)

        # Predict and detect anomalies
        predictions = model.predict(X_preprocessed)
        threshold = 0.0023196320236421467
        mse = np.mean(np.power(X_preprocessed - predictions, 2), axis=1)
        anomalies = mse > threshold

        # Create a DataFrame with the first 5 columns and index
        #saving the 5 initial columns of the dataframe for the ouput
        df_new = df.iloc[:, :5]
        df_new['Anomaly'] = ["Attack detected" if anomaly else "Normal" for anomaly in anomalies]
    
        num_anomalies = np.sum(anomalies)

        #saving df_new as a csv so that it can be fetched by another function
        df_new.to_csv('Anomaly_output.csv', index=False)

       # Generate and save the count plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_new, x='Anomaly')
        plt.title('Anomaly Detection Results')
        plt.xlabel('Connection type')
        plt.ylabel('Count')
       # Define y ticks at certain intervals
        max_count = df_new['Anomaly'].value_counts().max()
        ytick_values = range(0, max_count, 1000)  # Define the desired tick locations
        plt.yticks(ytick_values)
        count_plot_path = 'assets/count_plot.png'
        plt.savefig(count_plot_path)
        plt.close()

       # Convert DataFrame to HTML table with index
        table_html = df_new.to_html(index=False, classes='styled-table')  # Disable default index

        # Convert X_processed to JSON format
        df_json = df.to_json(orient='records')
        
        return render_template('anomaly_detection.html', table=table_html, num_anomalies=num_anomalies,df_json=df_json,count_plot_path=count_plot_path) 

    else:
        return 'Invalid file format'


def map_to_attack_types(predicted_labels):
    # Define the mapping of labels to attack types
    attack_types = {
        0: 'Dos',
        1: 'Probe',
        2: 'R2L',
        3: 'U2R',
        4: 'Normal'
    }
    
    # Map predicted labels to attack types
    attack_types_predicted = [attack_types[label] for label in predicted_labels]
    
    return attack_types_predicted


@app.route('/show-attack-types', methods=['POST'])
def show_attack_types():
    if request.method == 'POST':

        show_attack_types = request.form.get('show_attack_types')

        if show_attack_types == 'yes':
            #fetching the json and converting into numpy array
            df_json = request.form.get('df')
            df = pd.DataFrame(json.loads(df_json))
            
           
            # Call the function to fetch and display attack types
            df_new = fetch_attack_types(df)  # Implement this function
              # Generate and save the count plot
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_new, x='Attack Type')
            plt.title('LSTM Multiclassification Results')
            plt.xlabel('Connection types')
            plt.ylabel('Count')
            # Define y ticks at certain intervals
            max_count = df_new['Attack Type'].value_counts().max()
            ytick_values = range(0, max_count, 1000)  # Define the desired tick locations
            plt.yticks(ytick_values)
            count_plot_path_2 = 'assets/count_plot_2.png'
            plt.savefig(count_plot_path_2)
            plt.close()
            table_html_lstm = df_new.to_html(index=False, classes='styled-table')  # Disable default index
            return render_template('attack_types.html', table=table_html_lstm)

        else:
            return 'Attack types not requested.'

    return 'Invalid request.'

def fetch_attack_types(df):

    # One hot encoding
    categorical_features = ['protocol_type', 'service', 'flag']
    X_categorical = df[categorical_features]
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X_categorical)
    df_encoded = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(input_features=categorical_features))
    df_encoded = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)
     
    #Adding dummy columns to match the expected input shape of the model during testing 
    #can be a reasonable approach especially if you want to maintain consistency in your 
    #data preprocessing pipeline. (suited for real life senerios)
    if df_encoded.shape[1] != 118:
        # Adding dummy columns filled with 0 until the number of columns becomes 118
        num_dummy_cols = 118 - df_encoded.shape[1]
        dummy_cols = pd.DataFrame(0, index=np.arange(df_encoded.shape[0]), columns=[f'dummy_{i}' for i in range(num_dummy_cols)])
        df_encoded = pd.concat([df_encoded, dummy_cols], axis=1)

    print("after adding dummy columns",df_encoded.shape)

    # Apply standard scaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)

    # Reshape the input data to match the LSTM model's input shape
    X_test_reshaped = df_scaled.reshape(df_scaled.shape[0], 1, df_scaled.shape[1])


    # Predict attack types using the LSTM model
    predictions = lstm_model.predict(X_test_reshaped)
    
    # Reshape the predicted labels to match the expected shape
    y_pred_reshaped = predictions.reshape((predictions.shape[0], predictions.shape[2]))
    
    # Convert predicted probabilities to class labels
    predicted_labels = np.argmax(y_pred_reshaped, axis=1)
    
    # Read df_new from the CSV file
    df_new = pd.read_csv('Anomaly_output.csv')

    attack_labels = map_to_attack_types(predicted_labels)
    

    df_new = df_new.iloc[:, :5]
    df_new['Attack Type'] = attack_labels
    
    
    return df_new

if __name__ == '__main__':
    app.run(debug=True, port=5000)
