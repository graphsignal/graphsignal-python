from abc import ABC, abstractmethod

class BaseRecorder(ABC):
    def __init__(self):
        pass

    def setup(self):
        pass

    def shutdown(self):
        pass

    @abstractmethod
    def on_trace_start(self, signal, context):
        pass

    @abstractmethod
    def on_trace_stop(self, signal, context):
        pass
