
class Agent(object):
    def __init__(self):
        self.worker_id = None
        self.start_ms = None
        self.api_key = None
        self.workload_name = None
        self.run_id = None
        self.node_rank = None
        self.local_rank = None
        self.world_rank = None
        self.debug_mode = None
        self.uploader = None
        self.process_reader = None
        self.nvml_reader = None
        self.params = None
