"""
Intelligent Traffic Route Planner
Streamlit entrypoint for local execution and Streamlit Community Cloud deployment.
"""

import subprocess
import sys
from traffic_planner import _running_inside_streamlit, main

if __name__ == "__main__":
    if _running_inside_streamlit():
        main()
    else:
        # If started via `python app.py`, launch Streamlit automatically
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
