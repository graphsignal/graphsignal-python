import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
import chromadb

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.chroma_recorder import ChromaRecorder
from graphsignal.proto_utils import find_tag, find_param, find_data_count, find_data_sample

logger = logging.getLogger('graphsignal')


class ChromaRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_record(self):
        recorder = ChromaRecorder()
        recorder.setup()
        proto = signals_pb2.Span()
        context = {}
        recorder.on_span_start(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_stop(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_read(proto, context, DEFAULT_OPTIONS)

        self.assertEqual(proto.libraries[0].name, 'Chroma')

    @patch.object(Uploader, 'upload_span')
    async def test_trace_add(self, mocked_upload_span):
        client = chromadb.Client()

        collection = client.get_or_create_collection('test_col')

        collection.upsert(
            embeddings=[[1.5, 2.9, 3.4], [9.8, 2.3, 2.9]],
            documents=["doc1", "doc2"],
            metadatas=[{"source": "my_source"}, {"source": "my_source"}],
            ids=["id1", "id2"],
            increment_index=True
        )

        collection.get(
            ids=["id1", "id2"]
        )

        collection.query(
            query_embeddings=[[1.1, 2.3, 3.2], [5.1, 4.3, 2.2]],
            n_results=2
        )

        collection.delete()

        upsert_proto = mocked_upload_span.call_args_list[0][0][0]
        get_proto = mocked_upload_span.call_args_list[1][0][0]
        query_proto = mocked_upload_span.call_args_list[2][0][0]
        delete_proto = mocked_upload_span.call_args_list[3][0][0]

        # upsert
        self.assertEqual(upsert_proto.libraries[0].name, 'Chroma')
        self.assertEqual(find_tag(upsert_proto, 'component'), 'Memory')
        self.assertEqual(find_tag(upsert_proto, 'operation'), 'chroma.collection.upsert')
        self.assertEqual(find_tag(upsert_proto, 'index'), 'test_col')
        self.assertEqual(find_param(upsert_proto, 'collection'), 'test_col')
        self.assertEqual(find_data_count(upsert_proto, 'embeddings', 'element_count'), 6.0)

        # get
        self.assertEqual(get_proto.libraries[0].name, 'Chroma')
        self.assertEqual(find_tag(get_proto, 'component'), 'Memory')
        self.assertEqual(find_tag(get_proto, 'operation'), 'chroma.collection.get')
        self.assertEqual(find_tag(get_proto, 'index'), 'test_col')
        self.assertEqual(find_param(get_proto, 'collection'), 'test_col')
        self.assertEqual(find_data_count(get_proto, 'documents', 'element_count'), 2.0)
        self.assertIsNotNone(find_data_sample(get_proto, 'documents'))

        # query
        self.assertEqual(query_proto.libraries[0].name, 'Chroma')
        self.assertEqual(find_tag(query_proto, 'component'), 'Memory')
        self.assertEqual(find_tag(query_proto, 'operation'), 'chroma.collection.query')
        self.assertEqual(find_tag(query_proto, 'index'), 'test_col')
        self.assertEqual(find_param(query_proto, 'collection'), 'test_col')
        self.assertEqual(find_data_count(query_proto, 'query_embeddings', 'element_count'), 6.0)
        self.assertEqual(find_data_count(get_proto, 'documents', 'element_count'), 2.0)
        self.assertIsNotNone(find_data_sample(get_proto, 'documents'))

        # delete
        self.assertEqual(delete_proto.libraries[0].name, 'Chroma')
        self.assertEqual(find_tag(delete_proto, 'component'), 'Memory')
        self.assertEqual(find_tag(delete_proto, 'operation'), 'chroma.collection.delete')
        self.assertEqual(find_tag(delete_proto, 'index'), 'test_col')
        self.assertEqual(find_param(delete_proto, 'collection'), 'test_col')
