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


class HuggingFaceCallbackTest(unittest.TestCase):
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
        import torch
        if not torch.cuda.is_available():
            return
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
        full_train_dataset = tokenized_datasets["train"]
        full_eval_dataset = tokenized_datasets["test"]

        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-cased", cache_dir='temp', num_labels=2)

        from transformers import TrainingArguments
        training_args = TrainingArguments("test_trainer", num_train_epochs=1)
            
        from transformers import Trainer
        from graphsignal.callbacks.huggingface import GraphsignalPTCallback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset)
        trainer.add_callback(GraphsignalPTCallback())

        trainer.train()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.endpoint_name, 'training_step')

        self.assertTrue(
            signal.agent_info.framework_profiler_type, 
            signals_pb2.AgentInfo.ProfilerType.HUGGING_FACE_PROFILER)

        self.assertEqual(
            signal.frameworks[-1].type,
            signals_pb2.FrameworkInfo.FrameworkType.HUGGING_FACE_FRAMEWORK)
        self.assertTrue(signal.frameworks[-1].version.major > 0)

        self.assertTrue(len(signal.op_stats) > 0)
