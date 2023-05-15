import logging
import sys
import time
import chromadb

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import patch_method, unpatch_method, instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class ChromaRecorder(BaseRecorder):
    def __init__(self):
        self._library = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'Chroma'
        parse_semver(self._library.version, chromadb.__version__)

        from chromadb.api.models.Collection import Collection
        for method_name in ['add', 'update', 'upsert']:
            instrument_method(Collection, method_name, f'chroma.collection.{method_name}', trace_func=self.trace_add)
        instrument_method(Collection, 'delete', 'chroma.collection.delete', trace_func=self.trace_delete)
        instrument_method(Collection, 'get', 'chroma.collection.get', trace_func=self.trace_get)
        instrument_method(Collection, 'query', 'chroma.collection.query', trace_func=self.trace_query)

    def shutdown(self):
        from chromadb.api.models.Collection import Collection
        for method_name in ['add', 'update', 'upsert']:
            uninstrument_method(Collection, method_name)
        uninstrument_method(Collection, 'delete')
        uninstrument_method(Collection, 'get')
        uninstrument_method(Collection, 'query')

    def _fill(self, span, collection):
        span.set_tag('component', 'Memory')
        span.set_tag('index', collection.name)
        span.set_param('collection', collection.name)

    def trace_add(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, [
            'collection',
            'ids',
            'embeddings', 
            'metadatas', 
            'documents',
            'increment_index'])

        self._fill(span, params['collection'])

        if 'increment_index' in params:
            span.set_param('increment_index', params['increment_index'])

        if 'documents' in params:
            span.set_data('documents', params['documents'])

        if 'embeddings' in params:
            span.set_data('embeddings', params['embeddings'], record_data_sample=False)

    def trace_delete(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, [
            'collection',
            'ids',
            'where',
            'where_document'])

        self._fill(span, params['collection'])

    def trace_get(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, [
            'collection',
            'ids',
            'where', 
            'limit', 
            'offset', 
            'where_document',
            'include'])

        self._fill(span, params['collection'])

        if ret:
            if 'embeddings' in ret:
                span.set_data('embeddings', ret['embeddings'], record_data_sample=False)
            if 'documents' in ret:
                span.set_data('documents', ret['documents'])

    def trace_query(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, [
            'collection',
            'query_embeddings',
            'query_texts', 
            'n_results', 
            'documents', 
            'where',
            'where_document',
            'include'])

        self._fill(span, params['collection'])

        if 'n_results' in params:
            span.set_param('n_results', params['n_results'])

        if 'query_embeddings' in params:
            span.set_data('query_embeddings', params['query_embeddings'], record_data_sample=False)

        if 'query_texts' in params:
            span.set_data('query_texts', params['query_texts'])

        if ret:
            if 'embeddings' in ret:
                span.set_data('embeddings', ret['embeddings'])
            if 'documents' in ret:
                span.set_data('documents', ret['documents'])
            if 'distances' in ret:
                span.set_data('distances', ret['distances'])

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
