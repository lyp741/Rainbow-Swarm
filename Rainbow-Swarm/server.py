import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os
import argparse
import msgpack
import io
from PIL import Image
from PIL import ImageOps
import threading
import numpy as np
import time
import datetime
import sys
import pickle
import torch
from agent import Agent
from memory import ReplayMemory


parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help='websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help='server ip')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='../../log/model_test/', type=str,
                    help='reward log file name')
parser.add_argument('--agent-count', '-c', default=9, type=int,
                    help='number of agent')
parser.add_argument('--mode-distribute', '-d', default=False, type=bool,
                    help='mode distribute')
parser.add_argument('--mode-evaluate', '-e', default=False, type=bool,
                    help='mode evaluate no learning')

# ------Rainbow
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str,
                    default='space_invaders', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3),
                    metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4,
                    metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=64,
                    metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C',
                    help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10,
                    metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10,
                    metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS',
                    help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(8000),
                    metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4,
                    metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3,
                    metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99,
                    metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(100),
                    metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1,
                    metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0001,
                    metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4,
                    metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32,
                    metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3),
                    metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000,
                    metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10,
                    metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500,
                    metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=25000,
                    metavar='STEPS', help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true',
                    help='Display screen (testing only)')

args = parser.parse_args()
args.device = torch.device('cpu')

agent_count = args.agent_count
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        # a = 1/0
        self.write(greeting + ', friendly user!')


class StatusHandler(tornado.websocket.WebSocketHandler):

    agent = Agent(args)
    mem = ReplayMemory(args, args.memory_capacity,agent_count)
    agent_initialized = False
    cycle_counter = 1
    rgb_image_count = 1
    depth_image_count = 0
    depth_image_dim = 0
    ir_count = 1
    ground_count = 0
    compass_count = 1
    target_count = 1
    priority_weight_increase = (
        1 - args.priority_weight) / (args.T_max - args.learn_start)

    if args.mode_distribute:
        thread_event = threading.Event()

    state_cnn = torch.zeros(4,agent_count,3,128,128)
    state_oth = torch.zeros(4,agent_count,11)
    T = 0
    def open(self):
        print("open")

    def on_close(self):
        print("close")

    def on_message(self, message):
        print("received message")
        self.received_message(message)

    def callback(self, count):
        self.write_message('{"inventoryCount":"%d"}' % count)

    def send_action(self, action):
        dat = msgpack.packb({"command": "".join(map(str, action))})
        self.write_message(dat, binary=True)

    def received_message(self, m):
        payload = m
        dat = msgpack.unpackb(payload,  encoding='utf-8')
        image = []
        depth = []
        agent_count = len(dat['image'])
        for i in range(agent_count):
            image.append(Image.open(io.BytesIO(bytearray(dat['image'][i]))))
            if (self.depth_image_count == 1):
                depth_dim = len(dat['depth'][0])
                temp = (Image.open(io.BytesIO(bytearray(dat['depth'][i]))))
                depth.append(np.array(ImageOps.grayscale(
                    temp)).reshape(self.depth_image_dim))

        if(self.ir_count == 1):
            ir = dat['ir']
            ir_dim = len(ir[0])
        else:
            ir = []
            ir_dim = 0

        if(self.ground_count == 1):
            ground = dat['ground']
            ground_dim = len(ground[0])
        else:
            ground = []
            ground_dim = 0

        if (self.compass_count == 1):
            compass = dat['compass']
            compass_dim = len(compass[0])
        else:
            compass = []
            compass_dim = 0

        if(self.target_count == 1):
            target = dat['target']
            target_dim = len(target[0])
        else:
            target = []
            target_dim = 0
        self.agent.agent_count = agent_count
        observation = {"image": image, "depth": depth, "ir": ir,
                       "ground": ground, "compass": compass, "target": target}
        reward = np.array(dat['reward'], dtype=np.float32)
        reward = torch.tensor(reward)
        end_episode = np.array(dat['endEpisode'], dtype=np.bool)
        print("get daze!")

        s_cnn = self.agent._observation_to_state_cnn(observation)
        self.state_cnn = torch.stack((self.state_cnn[1],self.state_cnn[2],self.state_cnn[3],s_cnn))
        s_cnn_ = torch.cat([self.state_cnn[n] for n in range(4)],dim=1)
        
        s_oth = self.agent._observation_to_state_other(observation)
        self.state_oth = torch.stack((self.state_oth[1],self.state_oth[2],self.state_oth[3],s_oth))
        s_oth_ = torch.cat([self.state_oth[n] for n in range(4)],dim=1)

        
        state = {'cnn':s_cnn_,'oth':s_oth_}
        action = self.agent.act(state)
        action_ = action.numpy()
        self.send_action(action_)
        print(action)
        
        # for i in range(1000):
        self.mem.append({'cnn':s_cnn,'oth':s_oth},action,reward,end_episode)
        if self.T > 1000:
            self.agent.learn(self.mem,self.T)
        self.T += 1
        if self.T % args.replay_frequency == 0:
            # self.agent.reset_noise()  # Draw a new set of noisy weights
            pass
        # Update target network
        if self.T % args.target_update == 0:
            self.agent.update_target_net()


class Application(tornado.web.Application):
    def __init__(self):

        handlers = [

            (r'/', IndexHandler),
            (r'/ws', StatusHandler),
        ]

        settings = {
            'template_path': 'templates',
            'static_path': 'static'
        }

        tornado.web.Application.__init__(self, handlers, **settings)


if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(8765)
    print("start")
    tornado.ioloop.IOLoop.instance().start()
