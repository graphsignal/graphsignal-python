import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from fraud_model import FraudModel
import schedule

import graphsignal


# Configure Graphsignal
graphsignal.configure(api_key='testkey', debug_mode=True)

# Get logging session for the model
sess = graphsignal.session(model_name='my_fraud_detection_model', deployment_name='production')


model = FraudModel().double().cpu()
model.load_state_dict(torch.load('fraud_model.pt'))

def job():
    df = pd.read_csv('../../test-data/fraud-detection/creditcard.csv')
    transactions = df.drop(['Time', 'Class'], axis=1)
    transactions['Amount'] = StandardScaler().fit_transform(transactions['Amount'].values.reshape(-1, 1))
    #print(transactions.head())
    #transactions['V3'].values[:] = 0
    #print(transactions.head())
    transactions = transactions.sample(n=1000)

    loss = nn.MSELoss()
    scores = []
    data_loader = DataLoader(transactions.to_numpy())
    with torch.no_grad():
        for input in data_loader:
            output = model(input)
            scores.append(loss(output, input).data.item())

    # Log prediction
    sess.log_prediction(input_data=transactions, output_data=scores)

    '''sess.log_event(
        description='Fraud detection error',
        attributes={'data_file': '~/Downloads/creditcard.csv', 'model_file': 'fraud_model.pt'},
        is_error=True)'''

    print('Job finished.')

job()
schedule.every(10).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
