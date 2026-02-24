import os


def add_bootstrap_to_pythonpath():
    bootstrap_dir = os.path.dirname(__file__)
    
    python_path = os.environ.get("PYTHONPATH", "")
    if python_path:
        os.environ["PYTHONPATH"] = os.pathsep.join([bootstrap_dir, python_path])
    else:
        os.environ["PYTHONPATH"] = bootstrap_dir
