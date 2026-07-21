# Copyright 2026, Yumeng Liu @ USTC
"""城市地铁出行路径规划 —— 启动入口"""

from pathlib import Path

from gui import MetroApp

if __name__ == "__main__":
    DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
    app = MetroApp(DATA_ROOT)
    app.run()
