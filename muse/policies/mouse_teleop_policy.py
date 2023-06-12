import threading
import time
from collections import namedtuple
from pynput import keyboard

import numpy as np
from attrdict import AttrDict
from attrdict.utils import get_with_default

from muse.envs.env_interfaces import VRInterface
from muse.policies.policy import Policy
from muse.utils.torch_utils import cat_any, to_numpy
from muse.utils.transform_utils import fast_euler2quat, quat2mat, rotation_matrix

import pygame



class MouseTeleopPolicy(Policy):


    def _init_params_to_attrs(self, params):
        # this will be the pose (euler)
        self.action_name = get_with_default(params, "action_name", "action")    
        self.curr_pose = None          
        super(MouseTeleopPolicy, self)._init_params_to_attrs(params)


    def reset_policy(self, **kwargs):
        self._done = False
        self._step = 0

    def _init_setup(self):
        super(MouseTeleopPolicy, self)._init_setup()


    def warm_start(self, model, observation, goal):
        pass

    def get_action(self, model, observation, goal, **kwargs):
        #self._set_robot_orientation(observation)
        if self.curr_pose is None:
            self.curr_pose = observation.state[0][0][:2]
        mouse_pressed = pygame.mouse.get_pressed()
        if mouse_pressed[0]:
            mos_x, mos_y = pygame.mouse.get_pos()
            #print(mos_x,mos_y, mouse_pressed)
            #print(observation.state)
            #print(observation.agent.position)
            #print(observation.block.position)
            #print(observation.goal_pose)
            # Read Sensor TODO TESTING
            agent_pos  = observation.state[0][0][:2]
            #t_pos = observation.state[0][0][2:4]
            target_pos = np.array([mos_x, mos_y])
            delta = target_pos-agent_pos
            #err = np.linalg.norm(delta)
            delta_action = 0.1*delta 
        else:
            delta_action = [0,0]

        time.sleep(0.05)
        self.curr_pose  += delta_action
        command = self.curr_pose 

        # postprocess
        return self._postproc_fn(model, observation, goal, AttrDict.from_dict({
            self.action_name: np.concatenate([command]),
            'keypoint': np.concatenate([command]),
            'policy_type': np.array([258]),  
            'policy_name': np.array(["mouse_teleop"]), 
        }).leaf_apply(lambda arr: arr[None]))

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        return self._done
    

