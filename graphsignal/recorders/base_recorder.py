from abc import ABC, abstractmethod

class BaseRecorder(ABC):
    def __init__(self):
        pass

    def setup(self):
        pass

    def shutdown(self):
        pass

    def on_span_start(self, proto, context, options):
        pass

    def on_span_stop(self, proto, context, options):
        pass

    def on_span_read(self, proto, context, options):
        pass

    def on_metric_update(self):
        pass
