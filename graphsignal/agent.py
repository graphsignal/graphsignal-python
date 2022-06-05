from graphsignal.uploader import Uploader
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader

class Agent:
    worker_id: str = None
    start_ms: int = None
    api_key: str = None
    workload_name: str = None
    run_id: str = None
    global_rank: int = None
    node_rank: int = None
    local_rank: int = None
    debug_mode: bool = None
    disable_fwk_profiler: bool = False
    uploader: Uploader = None
    process_reader: ProcessReader = None
    nvml_reader: NvmlReader = None
    tags: dict = None
    params: dict = None
    metrics: dict = None
