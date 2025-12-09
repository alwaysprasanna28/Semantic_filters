import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.devices import Keyboard
from robosuite.wrappers import VisualizationWrapper

from semantic_filters import SemanticSafetyFilter
from cbf_viz import CBFRealtimePlot  
from robosuite.environments.base import register_env

try:
    from obj_env import OBJEnv
    register_env(OBJEnv)
    print(">> Successfully registered Adding OBJEnv!")
except ImportError:
    pass

if __name__ == "__main__":
    
    # 1. Config
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    controller_config["right"] = {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [0.03]*3 + [0.3]*3, "output_min": [-0.03]*3 + [-0.3]*3,
        "kp": 100, "damping_ratio": 1.5, "impedance_mode": "fixed",
        "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
        "uncouple_pos_ori": True, "input_type": "delta", "input_ref_frame": "base", "ramp_ratio": 0.2
    }

    # 2. Env
    env = suite.make(
        env_name="OBJEnv", robots="Panda", controller_configs=controller_config,
        has_renderer=True, has_offscreen_renderer=False, render_camera="agentview",
        ignore_done=True, use_camera_obs=False, control_freq=20,
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    robot = env.robots[0]
    # import ipdb; ipdb.set_trace()
    grip_site_name = robot.gripper["right"].important_sites["grip_site"]
    site_id = env.sim.model.site_name2id(grip_site_name)
    

    # 3. Filter
    safety_filter = SemanticSafetyFilter(env, robot_idx=0)
    
    # 4. Input
    device = Keyboard(env=env, pos_sensitivity=1.0, rot_sensitivity=1.0)
    env.viewer.add_keypress_callback(device.on_press)
    
    print("\n>> TELEOP READY. Click Window. Keys: Arrows (XY), . ; (Z)")
    viz = CBFRealtimePlot(x_range=(-.5,.5), y_range=(0, 1),rotate_90=True)
    
    
    obs_center = np.array([0, 0, 0]) 
    obs_scale = np.array([0.1, 0.1, 0.1])
    obs_eps = np.array([0.5, 0.5])
    while True:
        env.reset()
        env.render()
        device.start_control()
        
        while True:
            # Input
            # import ipdb; ipdb.set_trace()
            input_ac_dict = device.input2action()
            if input_ac_dict is None: break
            # import ipdb; ipdb.set_trace()
            raw_pos = input_ac_dict.get("right_delta", input_ac_dict.get("delta", np.zeros(3)))
            raw_ori = input_ac_dict.get("right_delta_ori", input_ac_dict.get("delta_ori", np.zeros(3)))
            raw_command = np.concatenate([raw_pos, raw_ori])
            gripper_val = input_ac_dict.get("right_gripper", [-1])
            is_grasping = gripper_val[0] > 0
            eef_pos = np.array(env.sim.data.site_xpos[site_id])
            eef_rot = np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3)
            # print(f"eef_pos: {eef_pos}")
            # Filter
            safe_command, status_str = safety_filter.solve(raw_command, is_grasping=is_grasping)
            
           
            bid, obj_pos, obj_rot = safety_filter._get_body_state("bluemug_main")
            
            if bid is not None:
                obj_dim = safety_filter._get_body_dimensions(bid)
                
                center, scale, epsilon = safety_filter.get_laptop_column_superquadric(obj_pos, obj_dim)
                
                viz.update_obstacle(center, scale, epsilon)
                viz.update_state(
                    eef_pos=eef_pos, 
                    raw_vel=raw_command, 
                    safe_vel=safe_command, 
                    velocity_scale=3.0  
                )
            # ----------------------

            try:
                active_robot = env.robots[0]
                grip_site = active_robot.gripper["right"].important_sites["grip_site"]
                sid = env.sim.model.site_name2id(grip_site)
                ee_pos = env.sim.data.site_xpos[sid]
                
                env.sim.model.site_pos[env.sim.model.site_name2id("vis_raw")] = ee_pos + (raw_command[:3]*5)
                env.sim.model.site_pos[env.sim.model.site_name2id("vis_safe")] = ee_pos + (safe_command[:3]*5)
            except: pass

            # Manual Action Map
            env_action = np.zeros(7)
            env_action[0:3] = safe_command[:3]
            env_action[3:6] = safe_command[3:]
            gripper_command = 1.0  
            env_action[6] = gripper_command if is_grasping else -1.0

        

            # if np.linalg.norm(raw_command[:3]) > 0.01:
            #     # print(f"\rStatus: {status_str: <25} | SafeCmd: {safe_command[:3]}", end="")

            env.step(env_action)
            env.render()