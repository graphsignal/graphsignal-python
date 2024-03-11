from abc import ABC, abstractmethod

class BaseRecorder(ABC):
    def __init__(self):
        pass

    def setup(self):
        pass

    def shutdown(self):
        pass

    def on_span_start(self, span, context):
        pass

    def on_span_stop(self, span, context):
        pass

    def on_span_read(self, span, context):
        pass

    def on_metric_update(self):
        pass
