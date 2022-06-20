import unittest
import logging
import sys
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class XGBoostCallbackTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_profile')
    def test_callback(self, mocked_upload_profile):
        import numpy as np
        import xgboost as xgb
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import dump_svmlight_file
        import joblib
        from sklearn.metrics import precision_score
        from graphsignal.profilers.xgboost import GraphsignalCallback

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

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertTrue(profile.step_stats.step_count > 0)
        self.assertTrue(profile.step_stats.total_time_us > 0)

        self.assertTrue(
            profile.profiler_info.framework_profiler_type, 
            profiles_pb2.ProfilerInfo.ProfilerType.XGBOOST_PROFILER)

        self.assertEqual(
            profile.frameworks[-1].type,
            profiles_pb2.FrameworkInfo.FrameworkType.XGBOOST_FRAMEWORK)
        self.assertTrue(profile.frameworks[-1].version.major > 0)

        self.assertEqual(len(profile.metrics), 2)

        self.assertTrue(len(profile.op_stats) > 0)
