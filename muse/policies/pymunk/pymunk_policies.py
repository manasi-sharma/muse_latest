import numpy as np
from attrdict.utils import get_with_default

from muse.envs.pymunk.slider_env_2d import SliderBlockEnv2D
from muse.policies.policy import Policy
from muse.utils.torch_utils import to_numpy

from attrdict import AttrDict as d

from muse.utils.transform_utils import unit_vector


class PymunkDirectPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # max speed by default
        self.speed = get_with_default(params, "speed", 0.04) #75.0) #1.0)
        self.noise_std = get_with_default(params, "noise_std", 0.)
        self.ego_speed = 0.04 #1 #0.04

    def _init_setup(self):
        assert isinstance(self._env, SliderBlockEnv2D) #and self._env.num_obstacles == 0

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, next_obs, next_goal, slider_x_center, slider_x_range):
        block_x_original = next_obs.block_positions[0, 0]
        block_y_original = next_obs.block_positions[0, 1]
        current_ego_pos = next_obs.position[0]

        # Pushing x block from above
        if current_ego_pos[1] < block_x_original[1]: # scenario where ego is below slider level
            self.path_type = 'below'
            self.crossed_1, self.crossed_2 = False, False
            offset_1 = np.array([np.random.uniform(32, 52), np.random.uniform(32, 62)])
            self.middle_point_1 = np.array([ slider_x_center[0] + slider_x_range[1] + offset_1[0] , block_x_original[1] - offset_1[1] ])
            offset_2 = np.array([np.random.uniform(0, 10), np.random.uniform(17, 52)])
            self.middle_point_2 = np.array([ self.middle_point_1[0] + offset_2[0] , slider_x_center[1] + offset_2[1] ])

        elif current_ego_pos[0] < block_x_original[0]:
            self.path_type = 'upper_left'

            self.crossed_1, self.crossed_2 = False, False
            offset_1 = np.array([np.random.uniform(32, 52), np.random.uniform(0, 10)])
            self.middle_point_1 = np.array([ block_x_original[0] + offset_1[0] , current_ego_pos[1] + offset_1[1] ])
            offset_2 = np.array([np.random.uniform(0, 10), 20])
            self.middle_point_2 = np.array([ self.middle_point_1[0] + offset_2[0] , block_x_original[1] + offset_2[1] ])
            
            """self.crossed = False
            offset = np.array([(block_x_original[0] - current_ego_pos[0]) + 20, 2])
            self.middle_point = np.array([block_x_original[0] + offset[0], block_x_original[1] + ((current_ego_pos[1] - block_x_original[1])/offset[1]) ])
            if abs(self.middle_point[0] - current_ego_pos[0]) < 32 or abs(self.middle_point[1] - current_ego_pos[1]) < 32:
                self.middle_point -= np.array([32, 32])"""

        else:
            self.path_type = 'upper_right'
            # self.crossed = None
            # self.middle_point = None

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)

        :return action: AttrDict (B x ...)
        """

        print("my body: ", observation.position)
        # print("block 1: ", self.world.bodies[0].position)
        # print("block 2: ", self.world.bodies[1].position)
        #print("self.slider_x_center: ", self.slider_x_center)
        #print("self.slider_x_range: ", self.slider_x_range)
        #print("block env: self.xleft: ", (self.slider_x_center + np.asarray([self.slider_x_range[0], 0]))[0])
        print("block env: block 1: ", observation.block_positions[0, 0])
        #print("middle 1: ", self.middle_point_1)
        #print("middle 2: ", self.middle_point_2)
        #print("self cross: ", self.crossed_1)
        #print("self.crossed: ", self.crossed_2)
        print()

        ego = to_numpy(observation.position, check=True).reshape(2)
        targ = to_numpy(observation.block_positions[:, :, [0], :], check=True).reshape(2)

        dist = np.linalg.norm(targ - ego)

        # clip the distance traveled
        #if self._env.ego_speed * self.speed > dist:
        """if self.ego_speed * self.speed > dist:
            # move the exact distance
            #vel = (targ - ego) / self._env.ego_speed
            vel = (targ - ego) / self.ego_speed
        else:
            # vector pointing to target, at self.speed
            vel = (targ - ego) / dist * self.speed"""
        
        """if self.complicated_path:
            if np.linalg.norm(self.middle_point - ego) < 1:
                self.crossed = True
            if self.crossed:
                vel = ((targ - ego) / np.linalg.norm(targ - ego))*10 #(targ - ego) / 5
            else:
                vel = ((self.middle_point - ego) / np.linalg.norm(self.middle_point - ego))*10 #(self.middle_point - ego) / 5
        else:
            vel = ((targ - ego) / np.linalg.norm(targ - ego))*10 #(targ - ego) / 5"""
        
        if self.path_type == 'below':
            if np.linalg.norm(self.middle_point_1 - ego) < 1:
                self.crossed_1 = True
            if np.linalg.norm(self.middle_point_2 - ego) < 1:
                self.crossed_2 = True
            
            if self.crossed_1 and not self.crossed_2:
                vel = ((self.middle_point_2 - ego) / np.linalg.norm(self.middle_point_2 - ego))*20
            elif self.crossed_1 and self.crossed_2:
                targ_top = targ + np.array([0, 20])
                vel = ((targ_top - ego) / np.linalg.norm(targ_top - ego))*20
            elif not self.crossed_1 and not self.crossed_2:
                vel = ((self.middle_point_1 - ego) / np.linalg.norm(self.middle_point_1 - ego))*20                               
            else:
                import pdb;pdb.set_trace()
                print("error")
            
        elif self.path_type == 'upper_left':

            if np.linalg.norm(self.middle_point_1 - ego) < 1:
                self.crossed_1 = True
            if np.linalg.norm(self.middle_point_2 - ego) < 1:
                self.crossed_2 = True
            
            if self.crossed_1 and not self.crossed_2:
                vel = ((self.middle_point_2 - ego) / np.linalg.norm(self.middle_point_2 - ego))*20
            elif self.crossed_1 and self.crossed_2:
                targ_top = targ + np.array([0, 20])
                vel = ((targ_top - ego) / np.linalg.norm(targ_top - ego))*20
            elif not self.crossed_1 and not self.crossed_2:
                vel = ((self.middle_point_1 - ego) / np.linalg.norm(self.middle_point_1 - ego))*20                               
            else:
                import pdb;pdb.set_trace()
                print("error")

            """if np.linalg.norm(self.middle_point - ego) < 1:
                self.crossed = True
            if self.crossed:
                vel = ((targ - ego) / np.linalg.norm(targ - ego))*20 #(targ - ego) / 5
            else:
                vel = ((self.middle_point - ego) / np.linalg.norm(self.middle_point - ego))*10 #(self.middle_point - ego) / 5"""
        
        else:
            targ_top = targ + np.array([0, 20])
            vel = ((targ_top - ego) / np.linalg.norm(targ_top - ego))*20 #(targ - ego) / 5

        # noise_std scaled to match environment noise_std
        if self.noise_std > 0.:
            #vel[:] += np.random.normal(0, self.noise_std, 2) / self._env.ego_speed
            vel[:] += np.random.normal(0, self.noise_std, 2) / self.ego_speed

        return d(
            action=vel[None],
        )