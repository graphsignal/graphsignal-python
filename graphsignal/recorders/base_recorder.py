from abc import ABC


class BaseRecorder(ABC):
    def __init__(self, root_pid=None, pid=None, args=None):
        # The watcher's target (root) process — the main server process
        # launched by graphsignal-run / graphsignal.watch(). `pid` is the
        # process this recorder observes: the target itself or one of its
        # descendant workers.
        self.root_pid = root_pid
        self.pid = pid
        self.args = args

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
