import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from matplotlib import pyplot as plt

class MultiGoalEnv(gym.Env):
    
    def __init__(self, nr_goal = 4, goal_reward=100.):
        radius = 2.
        self.min_x = -radius*1.1
        self.max_x = radius*1.1
        self.min_y = -radius*1.1
        self.max_y = radius*1.1

        self.state_shape = (1,2)
        self.low = np.array([self.min_x, self.min_y])
        self.high = np.array([self.max_x, self.max_y])

        self.dynamics = PointDynamics(dim=2, sigma=0.03)
        self.init_mu = np.array((0, 0), dtype=np.float32)
        self.init_sigma = 0.
        self.max_time_step = 20
        
        self.nr_goal = nr_goal
        if nr_goal == 1:
            goal_positions = np.array(
                [
                    [radius, 0]
                ],
                dtype=np.float32
            )
            avo_positions = np.array(
                [
                    [-radius*2/3,0]
                ],
                dtype=np.float32
            )
        elif nr_goal == 2:
            goal_positions = np.array(
                [
                    [radius, 0],
                    [-radius, 0]
                ],
                dtype=np.float32
            )
            avo_positions = np.array(
                [
                    [0, radius*2/3],
                    [0, -radius*2/3]
                ],
                dtype=np.float32
            )
        elif nr_goal == 3:
            goal_positions = np.array(
                [
                    [radius, 0],
                    [-radius/2, radius*np.sqrt(3)/2],
                    [-radius/2, -radius*np.sqrt(3)/2]
                ],
                dtype=np.float32
            )
            avo_positions = np.array(
                [
                    [radius*np.sqrt(3)/3,radius/3],
                    [radius*np.sqrt(3)/3,-radius/3],
                    [0, -radius*2/3]
                ],
                dtype=np.float32
            )
        elif nr_goal == 4:
            goal_positions = np.array(
                [
                    [radius, 0],
                    [0, radius],
                    [-radius, 0],
                    [0, -radius]
                ],
                dtype=np.float32
            )
            avo_positions = np.array(
                [
                    [radius*2/3/np.sqrt(2), radius*2/3/np.sqrt(2)],
                    [-radius*2/3/np.sqrt(2), radius*2/3/np.sqrt(2)],
                    [-radius*2/3/np.sqrt(2), -radius*2/3/np.sqrt(2)],
                    [radius*2/3/np.sqrt(2), -radius*2/3/np.sqrt(2)]
                ],
                dtype=np.float32
            )
        
        self.avo_positions = avo_positions
        self.goal_positions = goal_positions
        self.cost_scale = .2
        
        self.goal_threshold = .25
        self.goal_reward = goal_reward*0.
        self.action_cost_coeff = 0.
        self.vel_bound = radius/self.max_time_step
        
        self.action_space = spaces.Box(-self.vel_bound,self.vel_bound,shape=(2,))
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self.observation = self.reset()
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        action = action.ravel()

        a_lb = self.action_space.low
        a_ub = self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation, action)
        o_lb = self.observation_space.low
        o_ub = self.observation_space.high
        next_obs = np.clip(next_obs, o_lb, o_ub)

        reward = self.compute_reward(self.observation, action)
        cur_position = self.observation
        dist_to_goal_list = [
            np.linalg.norm(cur_position - goal_position)
            for goal_position in self.goal_positions
        ]
        dist_to_goal = np.amin(dist_to_goal_list)
        
        self.time_step += 1
        gaol_id = -1
#         done = dist_to_goal < self.goal_threshold
#         if done:
#             gaol_id = np.argmin(dist_to_goal_list)
#             reward += self.goal_reward
#         else:
        done = self.time_step > self.max_time_step
        if dist_to_goal < self.goal_threshold:
            gaol_id = np.argmin(dist_to_goal_list)
        
        self.observation = np.copy(next_obs)
        return next_obs, reward, done, {'pos':next_obs,'goal_id': gaol_id}

    def _reset(self):
        self.time_step = 0
        unclipped_observation = self.init_mu + self.init_sigma * \
            np.random.normal(size=self.dynamics.s_dim)
        o_lb = self.observation_space.low
        o_ub = self.observation_space.high
        self.observation = np.clip(unclipped_observation, o_lb, o_ub)
        return self.observation

    def render(self):

        delta = 0.01
        x_min, y_min  = tuple(1.1 * np.array(self.low))
        x_max, y_max = tuple(1.1 * np.array(self.high))
        
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        reward = 200*np.exp(-0.5*np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)/2/self.cost_scale) - 100*np.exp(-0.5*np.amin([
            (X - avo_x) ** 2 + (Y - avo_y) ** 2
            for avo_x, avo_y in self.avo_positions
        ], axis=0)/self.cost_scale)
        
        contours = plt.contour(X, Y, reward, 40)
        plt.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        goal = plt.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ko')
        avo = plt.plot(self.avo_positions[:, 0],
                       self.avo_positions[:, 1], 'kx')
        plt.xlim([x_min,x_max])
        plt.ylim([y_min,y_max])
        plt.grid(True)
        
    def plot_path(self,env_info_list, style='b'):
        self.render()
        path = np.concatenate([i['pos'][None] for i in env_info_list], axis=0)
        xx = path[:, 0]
        yy = path[:, 1]
        line, = plt.plot(xx, yy, style)

    def plot_paths(self,paths):
        self.render()
        line_lst = []
        for path in paths:
            positions = path["env_infos"]["pos"]
            xx = positions[:, 0]
            yy = positions[:, 1]
            line_lst += plt.plot(xx, yy, 'b')

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        # noinspection PyTypeChecker
        goal_reward = 200*np.exp(-0.5*np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])/2/self.cost_scale) - 100*np.exp(-0.5*np.amin([
            np.sum((cur_position - avo_position) ** 2)
            for avo_position in self.avo_positions
        ])/self.cost_scale)

        # penalize staying with the log barriers
        reward = np.sum([-action_cost, goal_reward])
        return reward

    def plot_position_cost(self):
        delta = 0.01
        x_min, y_min  = tuple(1.1 * np.array(self.low))
        x_max, y_max = tuple(1.1 * np.array(self.high))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_costs = 10*np.min([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ],axis=0) - self.reward_bias
        costs = goal_costs
        return [costs, self.goal_positions]

class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next