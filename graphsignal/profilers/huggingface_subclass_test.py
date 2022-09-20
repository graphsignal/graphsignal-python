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


class HuggingFaceSubclassTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @unittest.skip("long test")
    def test_subclass(self):
        import torch
        from datasets import load_dataset
        raw_datasets = load_dataset("imdb")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir='temp')

        def tokenize_function(examples):
            return tokenizer(examples["text"],
                             padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        small_train_dataset = tokenized_datasets["train"].shuffle(
            seed=42).select(
            range(10))
        small_eval_dataset = tokenized_datasets["test"].shuffle(
            seed=42).select(
            range(10))

        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-cased", cache_dir='temp', num_labels=2)

        from transformers import TrainingArguments
        training_args = TrainingArguments("test_trainer",
            num_train_epochs=1)
            
        from transformers import Trainer
        from graphsignal.profilers.pytorch import PyTorchProfiler

        profiler = PyTorchProfiler()
        signal = None

        class MyTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def prediction_step(self, *args, **kwargs):
                global signal
                signal = signals_pb2.WorkerSignal()
                profiler.start(signal)
                ret = super().prediction_step(*args, **kwargs)
                profiler.stop(signal)
                return ret

        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset)

        trainer.train()

        trainer.evaluate()

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        test_op_stats = None
        for op_stats in signal.op_stats:
            if 'sgemm' in op_stats.op_name:
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        import torch
        if torch.cuda.is_available():
            self.assertTrue(test_op_stats.total_device_time_us >= 1)
            self.assertTrue(test_op_stats.self_device_time_us >= 1)
        else:
            self.assertTrue(test_op_stats.total_host_time_us >= 1)
            self.assertTrue(test_op_stats.self_host_time_us >= 1)
