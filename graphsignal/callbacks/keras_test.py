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


class KerasCallbackTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @unittest.skip("enable manually")
    @patch.object(Uploader, 'upload_profile')
    def test_callback(self, mocked_upload_profile):
        import tensorflow as tf

        import tensorflow_datasets as tfds
        tfds.disable_progress_bar()

        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        ds_train = ds_train.map(normalize_img)
        ds_train = ds_train.batch(128)

        ds_test = ds_test.map(normalize_img)
        ds_test = ds_test.batch(128)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy']
        )

        from graphsignal.callbacks.keras import GraphsignalCallback

        model.fit(ds_train,
                  epochs=2,
                  validation_data=ds_test,
                  callbacks=[GraphsignalCallback()])

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        # pp.pprint(MessageToJson(profile))

        test_op_stats = None
        for op_stats in profile.op_stats:
            if op_stats.op_type == 'MatMul':
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.assertTrue(test_op_stats.total_device_time_us >= 1)
            self.assertTrue(test_op_stats.self_device_time_us >= 1)
        else:
            self.assertTrue(test_op_stats.total_host_time_us >= 1)
            self.assertTrue(test_op_stats.self_host_time_us >= 1)
