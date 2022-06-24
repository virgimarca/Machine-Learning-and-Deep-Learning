"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy
from re import A

import numpy as np
import gym
from gym import utils
from classes.env.mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):

    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0
        self.dr = False

    def set_random_parameters(self):
        if self.distribution == 'uniform':
            low_mean = 100
            high_mean = 1000
            std = (high_mean-low_mean)/10
            treshold_mass = 2

            length = len(self.sim.model.body_mass[1:])
            masses = np.zeros(length,dtype=np.float32)
            a = np.random.normal(low_mean,std)
            b = np.random.normal(high_mean,std)
            while a <= 0 or a > b:
              a = np.random.normal(low_mean,std)
              b = np.random.normal(high_mean,std)

            for i in range(length):
              if i == 0:
                masses[i] = 1
              else:
                  masses[i] = np.random.uniform(a,b)
                  while masses[i] < treshold_mass:
                    masses[i] = np.random.uniform(a,b)
            return masses
        elif self.distribution == 'truncnormal':
            l = []
            # because we keep the first mass fixes to 2.53429174
            l.append(2.53429174)
            for i, mean in enumerate(self.mu_dropo):
                std = self.cov_dropo[i]
                LB = 2
                UB = 10

                csi = truncnorm.rvs(-2, 2, loc=mean, scale=std)
                if (csi < LB) or (csi > UB):
                    while (csi < LB) or (csi > UB):
                        sample = truncnorm.rvs(-2, 2, loc=mean, scale=std)
                    csi = sample
                l.append(csi)
            return np.array(l).T
        else:
            print('The specified distribution for the domain randomization does not exist')

    def set_dropo_parameters(self, res):
        self.mu_dropo = res[0][0]
        self.cov_dropo = res[0][1]

    def set_distribution(self, distribution):
        self.distribution = distribution
        self.dr = True

    def sample_parameters(self):
        masses = self.set_random_parameters()
        self.set_parameters(masses)
        return

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        if self.dr:
          self.set_parameters(self.set_random_parameters())
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)
