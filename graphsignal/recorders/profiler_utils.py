import logging
import os
import tempfile
import gzip
import shutil
import glob

logger = logging.getLogger('graphsignal')

def create_log_dir():
    log_dir = tempfile.mkdtemp(prefix='graphsignal-')
    logger.debug('Created temporary log directory %s', log_dir)
    return log_dir

def remove_log_dir(log_dir):
    shutil.rmtree(log_dir)
    logger.debug('Removed temporary log directory %s', log_dir)

def find_and_read(log_dir, file_pattern, decompress=True, max_size=None):
    file_paths = glob.glob(os.path.join(log_dir, file_pattern))
    if len(file_paths) == 0:
        logger.debug('Files are not found at %s', os.path.join(log_dir, file_pattern))
        return None

    found_path = file_paths[-1]

    if max_size:
        file_size = os.path.getsize(found_path)
        if file_size > max_size:
            raise Exception('File is too big: {0}'.format(file_size))

    if decompress and found_path.endswith('.gz'):
        last_file = gzip.open(found_path, "rb")
    else:
        last_file = open(found_path, "rb")
    data = last_file.read()
    last_file.close()

    return data