import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import new_element

# Imports for your objects
try:
    # from laptop_ob import LaptopObject
    from semantic_filters.objects.bluemug_ob import BlueMugObject
    from semantic_filters.objects.can_ob import CanObject
except ImportError:
    pass 

class OBJEnv(ManipulationEnv):
    def __init__(self, robots, **kwargs):
        # Default Settings
        kwargs.setdefault("has_renderer", True)
        kwargs.setdefault("has_offscreen_renderer", False)
        kwargs.setdefault("ignore_done", True)
        kwargs.setdefault("control_freq", 20)
        
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()

        # 1. Setup Robot
        self.robots[0].robot_model.set_base_xpos([0, 0, 0])

        # 2. Setup Table
        self.mujoco_arena = TableArena(table_full_size=(0.8, 1.8, 0.8), table_friction=(1.0, 5e-3, 1e-4))
        self.mujoco_arena.set_origin([0.8, 0, 0])

        # 3. Setup Objects
        self.can_ob = CanObject()
        self.bluemug = BlueMugObject()
        
        # Position objects manually
        # self.laptop.get_obj().set("pos", "0.8 0 1")
        # self.can_ob.get_obj().set("quat", "0.7071 0 -0.7071 0")
        self.can_ob.get_obj().set("pos", "0.5 0.2 0.88")
        
        self.bluemug.get_obj().set("pos","0.8 0 0.8")

        # 4. Create Task
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.can_ob, self.bluemug], 
        )

      

    def _setup_references(self):
        super()._setup_references()
    
    def _setup_observables(self):
        return super()._setup_observables()

    def _reset_internal(self):
        super()._reset_internal()
    def reward(self, action=None):
        return 0.0
    


# obj_class = OBJEnv()
# print(obj_class.get_observables(robots="Panda"))