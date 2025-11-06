import logging
import os
from datetime import datetime

log_dir = "Drone_Logs"

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# handlers per file
hover_file = logging.FileHandler(
    os.path.join(log_dir, f"hover_logs_{run_id}.log"),
    mode='w'
)

path_file = logging.FileHandler(
    os.path.join(log_dir, f"path_logs_{run_id}.log"),
    mode='w'
)

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
hover_file.setFormatter(fmt)
path_file.setFormatter(fmt)

hover_logger = logging.getLogger("hover")
hover_logger.setLevel(logging.DEBUG)
hover_logger.addHandler(hover_file)

path_logger = logging.getLogger("path")
path_logger.setLevel(logging.DEBUG)
path_logger.addHandler(path_file)
