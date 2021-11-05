#Based on https://github.com/davidreiman/pytorch-atari-dqn

import os
import re
import gym
import time
import copy
import random
import warnings
import numpy as np

import torch
import torchvision
import torch.nn as nn
import cv2

from tqdm import tqdm_notebook as tqdm
from collections import deque, namedtuple

class DeepQNetwork(nn.Module):
    def __init__(self, num_frames, num_actions):
        super(DeepQNetwork, self).__init__()
        self.num_frames = num_frames
        self.num_actions = num_actions
        
        # Layers
        self.conv1 = nn.Conv2d(
            in_channels=num_frames,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
            )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
            )
        self.fc1 = nn.Linear(
            in_features=4160,
            out_features=256,
            )
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=num_actions,
            )
        
        # Activation Functions
        self.relu = nn.ReLU()
    
    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x
    
    def forward(self, x):
        
        # Forward pass
        x = self.relu(self.conv1(x))  # In: (80, 80, 4)  Out: (20, 20, 16)
        x = self.relu(self.conv2(x))  # In: (20, 20, 16) Out: (10, 10, 32)
        x = self.flatten(x)           # In: (10, 10, 32) Out: (3200,)
        x = self.relu(self.fc1(x))    # In: (3200,)      Out: (256,)
        x = self.fc2(x)               # In: (256,)       Out: (4,)
        
        return x

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'terminal', 'next_state'])

class Agent:
    def __init__(self, model, memory_depth, lr, gamma, epsilon_i, epsilon_f, anneal_time, ckptdir):
        
        self.cuda = True if torch.cuda.is_available() else False
        
        self.model = model
        self.device = torch.device("cuda" if self.cuda else "cpu")
        
        if self.cuda:
            self.model = self.model.cuda()
        
        self.memory_depth = memory_depth
        self.gamma = torch.tensor([gamma], device=self.device)
        self.e_i = epsilon_i
        self.e_f = epsilon_f
        self.anneal_time = anneal_time
        self.ckptdir = ckptdir
        
        if not os.path.isdir(ckptdir):
            os.makedirs(ckptdir)
        
        self.memory = deque(maxlen=memory_depth)
        self.clone()
        
        self.loss = nn.SmoothL1Loss()
        self.opt = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.95, eps=0.01)
        
    def clone(self):
        try:
            del self.clone_model
        except:
            pass
        
        self.clone_model = copy.deepcopy(self.model)
        
        for p in self.clone_model.parameters():
            p.requires_grad = False
        
        if self.cuda:
            self.clone_model = self.clone_model.cuda()
    
    def remember(self, *args):
        self.memory.append(Transition(*args))
    
    def retrieve(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, terminal, next_state = map(torch.cat, [*batch])
        return state, action, reward, terminal, next_state
    
    @property
    def memories(self):
        return len(self.memory)
    
    def act(self, state):
        q_values = self.model(state).detach()
        action = torch.argmax(q_values)
        return action.item()
    
    def process(self, state):

        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(80,105))
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=self.device, dtype=torch.float)
    
    def exploration_rate(self, t):
        if 0 <= t < self.anneal_time:
            return self.e_i - t*(self.e_i - self.e_f)/self.anneal_time
        elif t >= self.anneal_time:
            return self.e_f
        elif t < 0:
            return self.e_i
    
    def save(self, t):
        save_path = os.path.join(self.ckptdir, 'model-{}'.format(t))
        torch.save(self.model.state_dict(), save_path)
    
    def load(self):
        ckpts = [file for file in os.listdir(self.ckptdir) if 'model' in file]
        steps = [int(re.search('\d+', file).group(0)) for file in ckpts]
        
        latest_ckpt = ckpts[np.argmax(steps)]
        self.t = np.max(steps)
        
        print("Loading checkpoint: {}".format(latest_ckpt))
        
        self.model.load_state_dict(torch.load(os.path.join(self.ckptdir, latest_ckpt)))
        
    def update(self, batch_size):
        self.model.zero_grad()

        state, action, reward, terminal, next_state = self.retrieve(batch_size)
        q = self.model(state).gather(1, action.view(batch_size, 1))
        qmax = self.clone_model(next_state).max(dim=1)[0]
        
        nonterminal_target = reward + self.gamma*qmax
        terminal_target = reward
        
        target = terminal.float()*terminal_target + (~terminal).float()*nonterminal_target
    
        loss = self.loss(q.view(-1), target)
        loss.backward()
        self.opt.step()

    def play(self, episodes, train=False, load=False, plot=False, render=False, verbose=False):
    
        self.t = 0
        metadata = dict(episode=[], reward=[])
        
        if load:
            self.load()

        try:
            progress_bar = tqdm(range(episodes), unit='episode')
            
            i = 0
            for episode in progress_bar:

                state = env.reset()
                state = self.process(state)
                
                done = False
                total_reward = 0

                while not done:

                    if render:
                        env.render()

                    while state.size()[1] < num_frames:
                        action = 1 # Fire

                        new_frame, reward, done, info = env.step(action)
                        new_frame = self.process(new_frame)

                        state = torch.cat([state, new_frame], 1)
                    
                    if train and np.random.uniform() < self.exploration_rate(self.t-burn_in):
                        action = np.random.choice(num_actions)

                    else:
                        action = self.act(state)

                    new_frame, reward, done, info = env.step(action)
                    new_frame = self.process(new_frame)

                    new_state = torch.cat([state, new_frame], 1)
                    new_state = new_state[:, 1:, :, :]
                    
                    if train:
                        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                        action = torch.tensor([action], device=self.device, dtype=torch.long)
                        done = torch.tensor([done], device=self.device, dtype=torch.uint8)
                        
                        self.remember(state, action, reward, done, new_state)

                    state = new_state
                    total_reward += reward
                    self.t += 1
                    i += 1
                    
                    if not train:
                        time.sleep(0.1)
                    
                    if train and self.t > burn_in and i > batch_size:

                        if self.t % update_interval == 0:
                            self.update(batch_size)

                        if self.t % clone_interval == 0:
                            self.clone()

                        if self.t % save_interval == 0:
                            self.save(self.t)

                    if self.t % 1000 == 0:
                        progress_bar.set_description("t = {}".format(self.t))

                metadata['episode'].append(episode)
                metadata['reward'].append(total_reward.cpu().numpy())

                if episode % 100 == 0 and episode != 0:
                    avg_return = np.mean(metadata['reward'][-100:])
                    print("Average return (last 100 episodes): {:.2f}".format(avg_return))

                if plot:
                    plt.scatter(metadata['episode'], metadata['reward'])
                    plt.xlim(0, episodes)
                    plt.xlabel("Episode")
                    plt.ylabel("Return")
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
            
            env.close()
            return metadata

        except KeyboardInterrupt:
            if train:
                print("Saving model before quitting...")
                self.save(self.t)
            
            env.close()
            return metadata


# Hyperparameters

batch_size = 32
update_interval = 4
clone_interval = int(1e4)
save_interval = int(1e5)
frame_skip = None
num_frames = 4
num_actions = 4
episodes = int(1e5)
memory_depth = int(1e4)
epsilon_i = 1.0
epsilon_f = 0.1
anneal_time = int(1e6)
burn_in = int(5e4)
gamma = 0.99
learning_rate = 2.5e-4

model = DeepQNetwork(num_frames, num_actions)
agent = Agent(model, memory_depth, learning_rate, gamma, epsilon_i, epsilon_f, anneal_time, 'ckpt')
env = gym.make('BreakoutDeterministic-v4')
metadata = agent.play(episodes, train=True, load=False, verbose=True)
