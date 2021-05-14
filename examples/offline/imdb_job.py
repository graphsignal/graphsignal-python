import time
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb

import schedule


import graphsignal


# Configure Graphsignal
graphsignal.configure(api_key='testkey', debug_mode=True)

# Get logging session for the model
sess = graphsignal.session(deployment_name='imdb_prod')


max_features = 20000
maxlen = 80
model = keras.models.load_model('imdb_model.h5')
(_, _), (x_test, _) = imdb.load_data(num_words=max_features)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_test = x_test[np.random.choice(x_test.shape[0], 1000, replace=False),:]

def job():
    scores = model.predict(x_test)

    # Log prediction
    sess.log_prediction(output_data=scores)

job()
schedule.every(10).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)