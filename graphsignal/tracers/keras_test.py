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


class KerasCallbackTest(unittest.TestCase):
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
        ds_train = ds_train.take(100)
        ds_train = ds_train.cache()
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.map(normalize_img)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.take(100)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

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

        from graphsignal.tracers.keras import GraphsignalCallback

        model.fit(ds_train,
            batch_size=128,
            epochs=1,
            validation_data=ds_test)

        model.evaluate(ds_test,
            batch_size=128,
            callbacks=[GraphsignalCallback(model_name='m1', batch_size=128)])

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.model_name, 'm1')
        self.assertTrue(len(signal.inference_stats.extra_counters['items'].buckets_sec), 1)

        self.assertTrue(
            signal.agent_info.framework_profiler_type, 
            signals_pb2.AgentInfo.ProfilerType.KERAS_PROFILER)

        self.assertEqual(
            signal.frameworks[-1].type,
            signals_pb2.FrameworkInfo.FrameworkType.KERAS_FRAMEWORK)
        self.assertTrue(signal.frameworks[-1].version.major > 0)

        test_op_stats = None
        for op_stats in signal.op_stats:
            if 'MatMul' in op_stats.op_type:
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
