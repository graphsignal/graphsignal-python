from abc import ABC, abstractmethod

class BaseRecorder(ABC):
    def __init__(self):
        pass

    def setup(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def shutdown(self):
        pass

    def on_tick(self):
        pass
