
class Agent(object):
    def __init__(self):
        self.run_id = None
        self.run_start_ms = None
        self.api_key = None
        self.workload_name = None
        self.debug_mode = None
        self.uploader = None
        self.span_scheduler = None
        self.profiler = None
        self.active_span = None
        self.host_reader = None
        self.nvml_reader = None
