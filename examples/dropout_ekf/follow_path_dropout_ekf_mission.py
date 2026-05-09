import os
import runpy


if __name__ == "__main__":
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "..", "ekf", "follow_path_dropout_ekf_mission.py"),
        run_name="__main__",
    )
