

class OperationProfiler():
    def start(self, profile, context):
        raise NotImplementedError()

    def stop(self, profile, context):
        raise NotImplementedError()
