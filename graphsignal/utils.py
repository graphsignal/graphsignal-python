import uuid
import hashlib
import random
import tempfile
import shutil
import logging

logger = logging.getLogger('graphsignal')

RANDOM_CACHE = [random.random() for _ in range(10000)]
idx = 0

def fast_rand():
    global idx
    idx = (idx + 1) % len(RANDOM_CACHE)
    return RANDOM_CACHE[idx]


def sanitize_str(val, max_len=250):
    if not isinstance(val, str):
        return str(val)[:max_len]
    else:
        return val[:max_len]


def sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def uuid_sha1(size=-1):
    return sha1(str(uuid.uuid4()), size)

def create_log_dir():
    log_dir = tempfile.mkdtemp(prefix='graphsignal-')
    logger.debug('Created temporary log directory %s', log_dir)
    return log_dir

def remove_log_dir(log_dir):
    shutil.rmtree(log_dir)
    logger.debug('Removed temporary log directory %s', log_dir)
