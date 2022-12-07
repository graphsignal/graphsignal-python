import unittest
import logging
import sys
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class XGBoostCallbackTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_signal')
    def test_callback(self, mocked_upload_signal):
        import numpy as np
        import xgboost as xgb
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import dump_svmlight_file
        import joblib
        from sklearn.metrics import precision_score
        from graphsignal.callbacks.xgboost import GraphsignalCallback

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'max_depth': 3,
            'eta': 0.3,
            'objective': 'multi:softprob',
            'eval_metric': ['auc'],
            'num_class': 3}
        num_round = 20

        bst = xgb.train(
            params, 
            dtrain, 
            num_round, 
            evals=[(dtrain, 'Train'), (dtest, 'Valid')],
            callbacks=[GraphsignalCallback()])

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.endpoint_name, 'iteration')
