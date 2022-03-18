import os
from model.magic_point import MagicPoint

work_folder = os.environ.get("workspaceFolder")
print(work_folder)

data = os.environ.get("TEST")
print(data)