import unittest
import logging
import sys
import os
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class PyTorchLightningTest(unittest.TestCase):
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
        from pytorch_lightning import LightningModule, Trainer
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, random_split
        from torchmetrics import Accuracy
        from torchvision import transforms
        from torchvision.datasets import MNIST
        from graphsignal.profilers.pytorch_lightning import GraphsignalCallback

        PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
        AVAIL_GPUS = min(1, torch.cuda.device_count())
        BATCH_SIZE = 256 if AVAIL_GPUS else 64

        class MNISTModel(LightningModule):
            def __init__(self):
                super().__init__()
                self.batch_size = BATCH_SIZE
                self.l1 = torch.nn.Linear(28 * 28, 10)
                self.train_acc = Accuracy()

            def forward(self, x):
                return torch.relu(self.l1(x.view(x.size(0), -1)))

            def training_step(self, batch, batch_nb):
                x, y = batch
                preds = self(x)
                loss = F.cross_entropy(preds, y)
                self.train_acc(preds, y)
                self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
                return loss

            def train_dataloader(self):
                train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
                train_loader = DataLoader(train_ds, batch_size=self.batch_size)
                return train_loader

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.02)

        mnist_model = MNISTModel()

        trainer = Trainer(
            #accumulate_grad_batches=1,
            #auto_scale_batch_size=True,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=AVAIL_GPUS,
            max_epochs=3,
            callbacks=[GraphsignalCallback(batch_size=mnist_model.batch_size)]
        )

        trainer.tune(mnist_model)

        trainer.fit(mnist_model)

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertTrue(profile.step_stats.step_count > 0)
        self.assertTrue(profile.step_stats.sample_count > 0)
        self.assertTrue(profile.step_stats.total_time_us > 0)
        self.assertEqual(profile.step_stats.batch_size, mnist_model.batch_size)

        self.assertEqual(profile.metrics[0].name, 'train_acc')
        self.assertTrue(profile.metrics[0].value > 0)

        test_op_stats = None
        for op_stats in profile.op_stats:
            if op_stats.op_name == 'aten::addmm':
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
