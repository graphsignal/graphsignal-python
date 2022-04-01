
class Agent(object):
    def __init__(self):
        self.run_id = None
        self.run_start_ms = None
        self.api_key = None
        self.workload_name = None
        self.debug_mode = None
        self.uploader = None
        self.process_reader = None
        self.nvml_reader = None
