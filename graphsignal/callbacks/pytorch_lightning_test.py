import unittest
import logging
import sys
import os
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class PyTorchLightningTest(unittest.TestCase):
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
        from pytorch_lightning import LightningModule, Trainer
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, random_split
        from torchmetrics import Accuracy
        from torchvision import transforms
        from torchvision.datasets import MNIST
        from graphsignal.callbacks.pytorch_lightning import GraphsignalCallback

        PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
        AVAIL_GPUS = min(1, torch.cuda.device_count())
        BATCH_SIZE = 256 if AVAIL_GPUS else 64

        class MNISTModel(LightningModule):
            def __init__(self):
                super().__init__()
                self.batch_size = BATCH_SIZE
                self.l1 = torch.nn.Linear(28 * 28, 10)
                self.train_acc = Accuracy()
                self.test_acc = Accuracy()

            def forward(self, x):
                return torch.relu(self.l1(x.view(x.size(0), -1)))

            def training_step(self, batch, batch_nb):
                x, y = batch
                preds = self(x)
                loss = F.cross_entropy(preds, y)
                self.train_acc(preds, y)
                self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
                return loss

            def test_step(self, batch, batch_nb):
                x, y = batch
                preds = self(x)
                loss = F.cross_entropy(preds, y)
                self.test_acc(preds, y)
                self.log('test_acc', self.test_acc, on_step=True, on_epoch=False)
                return loss

            def train_dataloader(self):
                train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
                train_ds = torch.utils.data.Subset(train_ds, torch.arange(1000))
                train_loader = DataLoader(train_ds, batch_size=self.batch_size)
                return train_loader

            def test_dataloader(self):
                test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
                test_ds = torch.utils.data.Subset(test_ds, torch.arange(1000))
                test_loader = DataLoader(test_ds, batch_size=self.batch_size)
                return test_loader

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.02)

        mnist_model = MNISTModel()

        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=AVAIL_GPUS,
            max_epochs=1,
            callbacks=[GraphsignalCallback()]
        )

        trainer.tune(mnist_model)

        trainer.fit(mnist_model)

        trainer.test(mnist_model)

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.endpoint_name, 'test_batch')

        self.assertTrue(
            signal.agent_info.framework_profiler_type, 
            signals_pb2.AgentInfo.ProfilerType.PYTORCH_LIGHTNING_PROFILER)

        self.assertEqual(
            signal.frameworks[-1].type,
            signals_pb2.FrameworkInfo.FrameworkType.PYTORCH_LIGHTNING_FRAMEWORK)
        self.assertTrue(signal.frameworks[-1].version.major > 0)

        self.assertEqual(
            signal.model_info.model_format,
            signals_pb2.ModelInfo.ModelFormat.PYTORCH_FORMAT)
        self.assertTrue(signal.model_info.model_size_bytes > 0)

        self.assertTrue(len(signal.op_stats) > 0)
