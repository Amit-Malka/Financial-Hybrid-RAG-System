import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ui.gradio_app import iface

if __name__ == "__main__":
    iface.launch()
