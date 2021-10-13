import logging
import numpy as np
from tensorflow import keras
import json
from flask import Flask
from flask import request

import graphsignal

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Configure Graphsignal
graphsignal.configure(api_key='testkey')

# Get logging session for the model
sess = graphsignal.session(deployment_name='mnist_prod')

model = keras.models.load_model('mnist_model.h5')

app = Flask(__name__)

@app.route('/predict_digit', methods = ['POST'])
def predict_digit():
    input = request.get_json()
    
    logger.debug('Received prediction request: %s', input)

    output = model.predict(input)

    # Log prediction
    sess.log_prediction(prediction=output[0])

    logger.debug('Returning prediction output: %s', input)
    return json.dumps(output.tolist())

app.run(port=8090)