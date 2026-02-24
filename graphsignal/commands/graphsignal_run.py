import sys
import os
import runpy
import shutil
import logging
import graphsignal
from graphsignal.bootstrap.utils import add_bootstrap_to_pythonpath

log = logging.getLogger(__name__)

USAGE = """
Run given Python application with Graphsignal observability.

Example:
  graphsignal-run app.py
"""


def _find_executable(command):
    if os.path.isfile(command):
        return command
    return shutil.which(command)


def main():
    add_bootstrap_to_pythonpath()
    
    if len(sys.argv) < 2:
        print("graphsignal-run: no command specified\n")
        print(USAGE)
        sys.exit(1)
    
    sys.argv.pop(0)
    command = sys.argv[0]
    
    if os.path.isfile(command):
        sys.argv.pop(0)
        
        graphsignal.configure()
        
        try:
            runpy.run_path(command, run_name='__main__')
        except Exception as e:
            log.error("error running script '%s': %s", command, e, exc_info=True)
            sys.exit(1)
        
        return
    
    executable = _find_executable(command)
    
    if executable:
        sys.argv.pop(0)
        
        graphsignal.configure()
        
        log.debug("program executable: %s", executable)
        log.debug("program args: %s", sys.argv)
        
        try:
            os.execl(executable, executable, *sys.argv)
        except PermissionError:
            print("graphsignal-run: permission error while launching '%s'" % executable)
            print("Did you mean `graphsignal-run python %s`?" % executable)
            sys.exit(1)
        except Exception as e:
            print("graphsignal-run: error launching '%s': %s" % (executable, e))
            log.error("error launching executable", exc_info=True)
            raise
    
    else:
        sys.argv.pop(0)
        
        graphsignal.configure()
        
        try:
            runpy.run_module(command, run_name='__main__')
        except Exception as e:
            print("graphsignal-run: failed to find executable or module '%s'" % command)
            log.error("error running module '%s': %s", command, e, exc_info=True)
            sys.exit(1)

