import re
import os
try:
    import resource
except ImportError:
    resource = None


VM_RSS_REGEXP = re.compile(r'VmRSS:\s+(\d+)\s+kB')
VM_SIZE_REGEXP = re.compile(r'VmSize:\s+(\d+)\s+kB')


def cpu_time():
    if resource:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_utime + rusage.ru_stime

    return None


def vm_rss():
    output = None
    try:
        f = open('/proc/{0}/status'.format(os.getpid()))
        output = f.read()
        f.close()
    except Exception:
        return None

    match = VM_RSS_REGEXP.search(output)
    if match:
        return int(float(match.group(1)))

    return None


def vm_size():
    output = None
    try:
        f = open('/proc/{0}/status'.format(os.getpid()))
        output = f.read()
        f.close()
    except Exception:
        return None

    match = VM_SIZE_REGEXP.search(output)
    if match:
        return int(float(match.group(1)))

    return None
