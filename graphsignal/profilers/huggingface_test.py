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


class HuggingFacePTCallbackTest(unittest.TestCase):
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
        import torch
        if not torch.cuda.is_available():
            return
        from datasets import load_dataset
        raw_datasets = load_dataset("imdb")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        def tokenize_function(examples):
            return tokenizer(examples["text"],
                             padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        small_train_dataset = tokenized_datasets["train"].shuffle(
            seed=42).select(
            range(100))
        small_eval_dataset = tokenized_datasets["test"].shuffle(
            seed=42).select(
            range(100))
        full_train_dataset = tokenized_datasets["train"]
        full_eval_dataset = tokenized_datasets["test"]

        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2)

        from transformers import TrainingArguments
        training_args = TrainingArguments("test_trainer", 
            #fp16=True,
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=4)
            
        from transformers import Trainer
        from graphsignal.profilers.huggingface import GraphsignalPTCallback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset)
        trainer.add_callback(GraphsignalPTCallback())

        trainer.train()

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertTrue(len(profile.params) > 0)

        self.assertTrue(profile.step_stats.step_count > 0)
        self.assertTrue(profile.step_stats.sample_count > 0)
        self.assertTrue(profile.step_stats.total_time_us > 0)
        #self.assertTrue(profile.step_stats.flop_count > 0)
        self.assertTrue(profile.step_stats.batch_size > 0)
        self.assertTrue(profile.step_stats.device_batch_size > 0)

        test_op_stats = None
        for op_stats in profile.op_stats:
            if op_stats.op_name == 'aten::mm':
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
