

class OperationProfiler():
    def is_ready(self):
        return True

    def read_info(self, signal):
        raise NotImplementedError()

    def start(self, signal):
        raise NotImplementedError()

    def stop(self, signal):
        raise NotImplementedError()
