
class Agent:
    def __init__(self):
        self.worker_id = None
        self.api_key = None
        self.workload_name = None
        self.global_rank = None
        self.node_rank = None
        self.local_rank = None
        self.debug_mode = None
        self.disable_op_profiler = False
        self.uploader = None
        self.process_reader = None
        self.nvml_reader = None
        self.current_run = None
