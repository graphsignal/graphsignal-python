

class OperationProfiler():
    def read_info(self, signal):
        raise NotImplementedError()

    def start(self, signal, context):
        raise NotImplementedError()

    def stop(self, signal, context):
        raise NotImplementedError()
