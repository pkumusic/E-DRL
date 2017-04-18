"""2-link Planar Arm."""

import numpy as np
import gym
import gym.spaces


class TwoLinkArmEnv(gym.core.Env):
    DOF = 2
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self,
                 Q=None,
                 R=None,
                 goal_q=None,
                 init_q=None,
                 init_dq=None,
                 dt=1e-3,
                 l1=.5,
                 l2=.75,
                 m1=.33,
                 m2=.55,
                 izz1=15.,
                 izz2=8.,
                 noise_free=True,
                 noise_mu=None,
                 noise_sigma=None):
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -np.pi, -np.inf, -np.inf]),
            high=np.array([np.pi, np.pi, np.inf, np.inf]))
        self.action_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]))

        if Q is None:
            self.Q = np.zeros((self.DOF * 2, self.DOF * 2))
            self.Q[:self.DOF, :self.DOF] = np.eye(self.DOF) * 1000.0
        else:
            self.R = R

        if R is None:
            self.R = np.eye(self.DOF) * 0.001
        else:
            self.R = R

        self.dt = dt
        self._goal_q = goal_q
        self.goal_dq = np.zeros(self.DOF)
        self.init_q = np.zeros(self.DOF) if init_q is None else init_q
        self.init_dq = np.zeros(self.DOF) if init_dq is None else init_dq

        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.izz1 = izz1
        self.izz2 = izz2

        self.K1 = ((1 / 3. * self.m1 + self.m2) * self.l1**2. +
                   1 / 3. * self.m2 * self.l2**2.)
        self.K2 = self.m2 * self.l1 * self.l2
        self.K3 = 1 / 3. * self.m2 * self.l2**2.
        self.K4 = 1 / 2. * self.m2 * self.l1 * self.l2

        # how much noise to add to input signal
        self.noise_free = noise_free
        self.noise_mu = np.zeros(
            (self.DOF, )) if noise_mu is None else noise_mu
        self.noise_sigma = np.ones(
            (self.DOF, )) if noise_sigma is None else noise_sigma

        self.reset()

        self.viewer = None

    def get_jacobian(self):
        jacobian = np.zeros((self.DOF, self.DOF))
        jacobian[0, 1] = self.l2 * -np.sin(self.q[0] + self.q[1])
        jacobian[1, 1] = self.l2 * np.cos(self.q[0] + self.q[1])
        jacobian[0, 0] = self.l1 * -np.sin(self.q[0]) + jacobian[0, 1]
        jacobian[1, 0] = self.l1 * np.cos(self.q[0]) + jacobian[1, 1]
        return jacobian

    def _reset(self):
        if self._goal_q is None:
            self.goal_q = (2 * np.pi) * np.random.rand(self.DOF) - np.pi
        else:
            self.goal_q = self._goal_q.copy()
        self.q = self.init_q.copy()
        self.dq = self.init_dq.copy()
        self.t = 0.

        return np.hstack((self.q, self.dq))

    @property
    def position(self):
        return np.copy(self.q)

    @property
    def velocity(self):
        return np.copy(self.dq)

    @property
    def state(self):
        return np.hstack((self.q, self.dq))

    @state.setter
    def state(self, value):
        self.q = value[:self.DOF, ...]
        self.dq = value[self.DOF:, ...]

    @property
    def goal(self):
        return np.hstack((self.goal_q, self.goal_dq))

    def _step(self, u, dt=None):
        if dt is None:
            dt = self.dt

        if not self.noise_free:
            u0_noise = np.random.normal(self.noise_mu[0], self.noise_sigma[0])
            u1_noise = np.random.normal(self.noise_mu[1], self.noise_sigma[1])
            u[0] += u0_noise
            u[1] += u1_noise

        u = np.clip(u, self.action_space.low, self.action_space.high)

        C2 = np.cos(self.q[1])
        S2 = np.sin(self.q[1])
        M11 = (self.K1 + self.K2 * C2)
        M12 = (self.K3 + self.K4 * C2)
        M21 = M12
        M22 = self.K3
        H1 = (-self.K2 * S2 * self.dq[0] * self.dq[1] -
              1 / 2.0 * self.K2 * S2 * self.dq[1]**2.0)
        H2 = 1 / 2. * self.K2 * S2 * self.dq[0]**2.

        ddq1 = ((H2 * M11 - H1 * M21 - M11 * u[1] + M21 * u[0]) /
                (M12**2. - M11 * M22))
        ddq0 = (-H2 + u[1] - M22 * ddq1) / M21

        self.dq += np.array([ddq0, ddq1]) * dt
        self.q += self.dq * dt
        self.t += dt

        # calculate the reward
        x_diff = np.hstack((self.q, self.dq)) - np.hstack(
            (self.goal_q, self.goal_dq))

        reward = -x_diff.dot(self.Q).dot(x_diff) - u.dot(self.R).dot(u)
        reward *= self.dt
        is_done = False
        if np.allclose(
                self.goal_q, self.q, atol=.01) and np.allclose(
                    self.goal_dq, self.dq, atol=.01):
            is_done = True

        return np.hstack((self.q, self.dq)), reward, is_done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        l, r, t, b = 0, 1, .1, -.1
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)

            max_arm_length = 2 * self.l1 + self.l2
            bounds = 1.5 * max_arm_length
            self.viewer.set_bounds(-bounds, bounds, -bounds, bounds)

        # add goal geoms
        link1_goal = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        link1_goal_transform = rendering.Transform(rotation=self.goal_q[0])
        link1_goal.add_attr(link1_goal_transform)
        link1_goal._color.vec4 = (1., 0., 0., 0.25)
        self.viewer.add_onetime(link1_goal)

        p1_goal = [
            2 * self.l1 * np.cos(self.goal_q[0]),
            2 * self.l1 * np.sin(self.goal_q[0])
        ]

        link2_goal = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        link2_goal_transform = rendering.Transform(
            rotation=self.goal_q[0] + self.goal_q[1],
            translation=tuple(p1_goal))
        link2_goal.add_attr(link2_goal_transform)
        link2_goal._color.vec4 = (0., 0., 1., 0.25)
        self.viewer.add_onetime(link2_goal)

        p1 = [2 * self.l1 * np.cos(self.q[0]), 2 * self.l1 * np.sin(self.q[0])]

        # add the arm geoms
        link1 = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link1_transform = rendering.Transform(rotation=self.q[0])
        link1.add_attr(link1_transform)
        link1.set_color(1., 0., 0.)

        link2 = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link2_transform = rendering.Transform(
            rotation=self.q[0] + self.q[1], translation=tuple(p1))
        link2.add_attr(link2_transform)
        link2.set_color(0., 0., 1.)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


class LimitedTorqueTwoLinkArmEnv(TwoLinkArmEnv):
    def __init__(self, max_torques=None, **kwargs):
        super(LimitedTorqueTwoLinkArmEnv, self).__init__(**kwargs)

        if max_torques is None:
            max_torques = np.array([10.0, 10.0])

        self.action_space = gym.spaces.Box(low=-max_torques, high=max_torques)
