from abc import ABC, abstractmethod

class BaseRecorder(ABC):
    def __init__(self):
        pass

    def setup(self):
        pass

    def shutdown(self):
        pass

    @abstractmethod
    def on_trace_start(self, proto, context, options):
        pass

    @abstractmethod
    def on_trace_stop(self, proto, context, options):
        pass

    @abstractmethod
    def on_trace_read(self, proto, context, options):
        pass

    @abstractmethod
    def on_metric_update(self):
        pass
