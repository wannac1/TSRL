# file: training/agents/__init__.py
# description: Makes the 'agents' directory a Python package and exposes key classes.

# 从相应文件中导入核心类，以便于从包的顶层直接访问
from .tutor_ppo import TutorPPO
from .state_manager import StateManager
from .detector_wrapper import DetectorWrapper
from .tutor_grpo import TutorGRPO

# 定义当使用 from training.agents import * 时，哪些模块会被导入
__all__ = [
    'TutorPPO',
    'StateManager',
    'DetectorWrapper',
    'TutorGRPO'
]