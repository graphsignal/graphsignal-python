

class BaseProfiler():
    def start(self):
        raise NotImplementedError()

    def stop(self, profile):
        raise NotImplementedError()
