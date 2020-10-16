"""Utility methods.
"""

import collections
import contextlib
import sys
import itertools
import imageio
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from settings import GeneralConfig as gc
from settings import EnvConfig as ec
from settings import ApproachConfig as ac


def flatten(x):
    """Flatten the (possibly irregular) input list.
    """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    return [x]


class FCN(nn.Module):
    """Fully connected network.
    """
    def __init__(self, in_size, hid_sizes, out_size, do_softmax=False):
        super().__init__()
        self.do_softmax = do_softmax
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears.append(nn.Linear(in_size, hid_sizes[0]))
        self.bns.append(nn.BatchNorm1d(num_features=hid_sizes[0]))
        for i in range(len(hid_sizes)-1):
            self.linears.append(nn.Linear(hid_sizes[i], hid_sizes[i+1]))
            self.bns.append(nn.BatchNorm1d(num_features=hid_sizes[i+1]))
        self.linears.append(nn.Linear(hid_sizes[-1], out_size))

    def forward(self, x):  # pylint:disable=arguments-differ
        if x.dim() == 1:
            # Add in dummy batch dimension.
            x = x.unsqueeze(dim=0)
        for i, linear in enumerate(self.linears[:-1]):
            x = self.bns[i](F.relu(linear(x)))
        x = self.linears[-1](x)
        if self.do_softmax:
            x = F.softmax(x, dim=1)
        return x


class CNN(nn.Module):
    """Convolutional network.
    """
    def __init__(self, in_size, num_channels, kernel_size, hid_sizes,
                 out_size, theta_shape, do_max_pool=False, do_softmax=False):
        super().__init__()
        self.other_size = in_size-np.prod(theta_shape)
        self.theta_shape = theta_shape
        assert len(theta_shape) == 3
        self.do_max_pool = do_max_pool
        self.do_softmax = do_softmax
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels=theta_shape[0],
                                    out_channels=num_channels,
                                    kernel_size=kernel_size))
        if do_max_pool:
            # max-pool with kernel and stride 4
            self.pools = nn.ModuleList()
            self.pools.append(nn.MaxPool2d(kernel_size=4))
            flattened_img_size = (num_channels*
                                  ((theta_shape[1]-kernel_size+1)//4)*
                                  ((theta_shape[2]-kernel_size+1)//4))
        else:
            flattened_img_size = (num_channels*
                                  (theta_shape[1]-kernel_size+1)*
                                  (theta_shape[2]-kernel_size+1))
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(self.other_size+flattened_img_size,
                                      hid_sizes[0]))
        for i in range(len(hid_sizes)-1):
            self.linears.append(nn.Linear(hid_sizes[i], hid_sizes[i+1]))
        self.linears.append(nn.Linear(hid_sizes[-1], out_size))

    def forward(self, x):  # pylint:disable=arguments-differ
        if x.dim() == 1:
            # Add in dummy batch dimension.
            x = x.unsqueeze(dim=0)
        other, theta = torch.split(x, (self.other_size,
                                       np.prod(self.theta_shape)), dim=1)
        theta = theta.reshape((theta.shape[0],)+self.theta_shape)
        theta = self.convs[0](theta)
        if self.do_max_pool:
            theta = self.pools[0](theta)
        theta = torch.flatten(theta, start_dim=1)
        x = torch.cat((other, theta), dim=1)
        for linear in self.linears[:-1]:
            x = F.relu(linear(x))
        x = self.linears[-1](x)
        if self.do_softmax:
            x = F.softmax(x, dim=1)
        return x


def powerset(iterable):
    """Return an iterable over the powerset of the given iterable.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s)+1))


def lin_interp(in_low, in_high, out_low, out_high, in_val):
    """Returns an out_val.
    """
    slope = (out_high-out_low)/(in_high-in_low)
    return out_low+slope*(in_val-in_low)


class DummyFile:
    """Helper for nostdout().
    """
    def write(self, x):
        """Dummy write method.
        """
        pass

    def flush(self):
        """Dummy flush method.
        """
        pass


@contextlib.contextmanager
def nostdout():
    """Context for suppressing output. Usage:
    import nostdout
    with nostdout():
        foo()
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class DebugInfoVisualizer:
    """Store text to display when rendering images or videos
    """
    _global_log = ""
    _episode_log = ""
    _step_log = ""

    @classmethod    
    def init(cls):
        cls._global_log = ""
        cls._episode_log = ""
        cls._step_log = ""

    @classmethod    
    def reset(cls):
        cls._episode_log = ""
        cls._step_log = ""

    @classmethod    
    def step(cls):
        if gc.verbosity > 2:
            print(cls._get_message())
        cls._step_log = ""

    @classmethod    
    def log_global_info(cls, s):
        cls._global_log += s

    @classmethod    
    def log_episode_info(cls, s):
        cls._episode_log += s

    @classmethod    
    def log_step_info(cls, s):
        cls._step_log += s

    @classmethod    
    def _get_message(cls):
        return cls._global_log + "\n" + cls._episode_log + "\n" + cls._step_log

    @classmethod
    def render(cls, image):
        text = cls._get_message()
        size = min(image.shape[:2])

        lines = text.split("\n")
        max_text_line = lines[np.argmax([len(s) for s in lines])]
        text_len = len(max_text_line)

        # Availability is platform dependent
        font = 'Arial'

        # Create font
        pil_font = ImageFont.truetype(font + ".ttf", size=2*size // text_len,
                                      encoding="unic")
        text_width, text_height = pil_font.getsize(max_text_line)
        text_height *= len(lines)

        # create a blank canvas with extra space between lines
        canvas = Image.new('RGBA', [image.shape[0], image.shape[0]], (255, 255, 255, 255))

        # draw the text onto the canvas
        draw = ImageDraw.Draw(canvas)
        offset = ((image.shape[0] - text_width) // 2,
                  (image.shape[0] - text_height) // 2)
        white = "#000000"
        draw.text(offset, text, font=pil_font, fill=white)

        # Convert the canvas into an array with values in [0, 1]
        out = (255 - np.asarray(canvas)) / 255.0
        out = np.array(255*out, dtype=np.uint8)

        # Concat to input image
        return np.hstack([image, out])


def get_trajectories(policy, env, num_episodes=1):
    """Execute the policy for a certain number of episodes in the environment
    """
    trajectories = []
    for _ in range(num_episodes):
        trajectory = []
        state = env.sample_initial_state()
        step = 0
        while True:
            action = policy(state)
            _, done = env.reward(state, action)
            trajectory.append((state, action))
            if done or step > ec.max_episode_length:
                break
            step += 1
            state = env.sample_next_state(state, action)
        trajectories.append(trajectory)
    return trajectories

def test_approach(env, approach, render=False, video_path=None, verbose=False,
                  train_or_test="test"):
    """Test the given approach in the given env.
    """
    DebugInfoVisualizer.init()
    reset_cost = approach.reset_test_environment(env)
    all_returns = []
    all_planning_costs = []
    all_objective_values = []
    images = []
    num_ep = (gc.num_eval_episodes_test if train_or_test == "test"
              else gc.num_eval_episodes_train)
    # env.transition_model._pause_pybullet(5)
    for episode in range(num_ep):  # loop over independent episodes
        episode_reset_cost = approach.reset_episode()
        planning_cost = reset_cost  # add this cost on every episode
        planning_cost += episode_reset_cost
        state = env.sample_initial_state()
        step = 0
        returns = 0

        DebugInfoVisualizer.reset()
        DebugInfoVisualizer.log_episode_info("Episode: {}\n".format(episode))

        if render:
            image = env.render(state)
            if gc.use_debug_info_visualizer:
                image = DebugInfoVisualizer.render(image)
            images.append(image)

        while True:
            DebugInfoVisualizer.step()

            action, step_cost = approach.get_action(state)
            planning_cost += step_cost
            rew, done = env.reward(state, action)
            returns += rew*(ec.gamma**step)

            DebugInfoVisualizer.log_step_info("Step: {}\n".format(step))
            DebugInfoVisualizer.log_step_info("Returns: {:06.3f}\n".format(returns))
            DebugInfoVisualizer.log_step_info("Action: {}\n".format(action))
            DebugInfoVisualizer.log_step_info("Step cost: {:06.3f}\n".format(step_cost))
            if render:
                image = env.render(state)
                if gc.use_debug_info_visualizer:
                    image = DebugInfoVisualizer.render(image)
                images.append(image)

            if done or step > ec.max_episode_length:
                if gc.verbosity > 2:
                    print(state, rew)
                break
            if action is None:
                break
            next_state = env.sample_next_state(state, action)
            if gc.verbosity > 2:
                print(state, action, rew, next_state)
            step += 1
            state = next_state
        if gc.verbosity > 2:
            print("finished episode with returns {:.5f}".format(returns))
            input("!!")
        all_returns.append(returns)
        all_planning_costs.append(planning_cost)
        objective_value = returns-ac.lam*planning_cost
        all_objective_values.append(objective_value)
        print("finished episode {} in {} steps with returns {}, planning cost "
              "{}, objective {}".format(episode, step, returns, planning_cost,
                                        objective_value), flush=True)
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    mean_planning_cost = np.mean(all_planning_costs)
    std_planning_cost = np.std(all_planning_costs)
    mean_objective_value = np.mean(all_objective_values)
    std_objective_value = np.std(all_objective_values)
    if verbose:
        print("Average discounted return was {:.5f} (std = {:.5f})".format(
            mean_return, std_return))
        print("Average planning cost was {:.5f} (std = {:.5f})".format(
            mean_planning_cost, std_planning_cost))
        print("Average objective value was {:.5f} (std = {:.5f})".format(
            mean_objective_value, std_objective_value))
    if render and not ec.family_to_run.startswith("tamp"):
        if video_path is None:
            video_path = "/tmp/{}_{}_{}.mp4".format(env.__class__.__name__,
                                                    approach.__class__.__name__,
                                                    gc.seed)
        imageio.mimwrite(video_path, images)
        print("Wrote out video to {}".format(video_path))
    return (mean_return, mean_planning_cost, mean_objective_value,
            std_return, std_planning_cost, std_objective_value)
