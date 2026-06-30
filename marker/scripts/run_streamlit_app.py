import subprocess
import os
import sys


def streamlit_app_cli(app_name: str = "streamlit_app.py"):
    argv = sys.argv[1:]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(cur_dir, app_name)
    
    # Add root directory to PYTHONPATH so streamlit can find the 'marker' package
    root_dir = os.path.dirname(os.path.dirname(cur_dir))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = root_dir + os.pathsep + existing_pythonpath if existing_pythonpath else root_dir
    
    cmd = [
        "streamlit",
        "run",
        app_path,
        "--server.fileWatcherType",
        "none",
        "--server.headless",
        "true",
    ]
    if argv:
        cmd += ["--"] + argv
    subprocess.run(cmd, env={**os.environ, "IN_STREAMLIT": "true", "PYTHONPATH": new_pythonpath})


def extraction_app_cli():
    streamlit_app_cli("extraction_app.py")
