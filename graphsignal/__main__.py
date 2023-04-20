import sys
import runpy
import graphsignal


def main():
    # remove own path
    sys.argv.pop(0)

    # detect if we are running as a module
    is_module = False
    if len(sys.argv) >= 1:
        if sys.argv[0] == '-m':
            is_module = True
            sys.argv.pop(0)
    else:
        sys.exit(1)

    if len(sys.argv) < 1:
        sys.exit(1)

    # initialize graphsignal   
    graphsignal.configure()

    # run it as normal
    module_or_script = sys.argv[0]
    if is_module:
        runpy.run_module(module_or_script, run_name='__main__')
    else:
        exec(open(module_or_script).read(), globals(), locals())


if __name__ == "__main__":
    main()
