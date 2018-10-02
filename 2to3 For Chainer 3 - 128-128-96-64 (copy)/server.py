# -*- coding: utf-8 -*-

import os
import cherrypy
import argparse
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from cnn_dqn_agent import CnnDqnAgent
import msgpack
import io
from PIL import Image
from PIL import ImageOps
import threading
import numpy as np
import time
import datetime
import sys
import pickle as pickle

# import cupy.cuda.runtime as rt

parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help='websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help='server ip')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='../../log/model_test/', type=str,
                    help='reward log file name')
parser.add_argument('--agent-count', '-c', default=1, type=int,
                    help='number of agent')
parser.add_argument('--mode-distribute', '-d', default=False, type=bool,
                    help='mode distribute')
parser.add_argument('--mode-evaluate', '-e', default=False, type=bool,
                    help='mode evaluate no learning')
parser.add_argument('--model', '-m', default='None', type=str,
                    help='model')
args = parser.parse_args()


class Root(object):
    @cherrypy.expose
    def index(self):
        return 'some HTML with a websocket javascript connection'

    @cherrypy.expose
    def ws(self):
        # you can access the class instance through
        handler = cherrypy.request.ws_handler


class AgentServer(WebSocket):
    agent = CnnDqnAgent()
    agent_initialized = False
    cycle_counter = 1
    rgb_image_count = 1
    depth_image_count = 0
    depth_image_dim = 0
    ir_count = 1
    ground_count = 0
    compass_count = 1
    target_count = 1

    if args.mode_distribute:
        thread_event = threading.Event()

    def send_action(self, action):
        dat = msgpack.packb({"command": "".join(map(str, action))})
        self.send(dat, binary=True)

    def received_message(self, m):
        payload = m.data
        dat = msgpack.unpackb(payload,  encoding='utf-8')

        image = []
        depth = []
        agent_count = len(dat['image'])
        
        for i in range(agent_count):
            image.append(Image.open(io.BytesIO(bytearray(dat['image'][i]))))
            if (self.depth_image_count == 1):
                depth_dim = len(dat['depth'][0])
                temp = (Image.open(io.BytesIO(bytearray(dat['depth'][i]))))
                depth.append(np.array(ImageOps.grayscale(temp)).reshape(self.depth_image_dim))

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
        
        observation = {"image": image, "depth":depth, "ir":ir, "ground":ground, "compass":compass, "target":target}
        reward = np.array(dat['reward'], dtype=np.float32)
        end_episode = np.array(dat['endEpisode'], dtype=np.bool)

        if not self.agent_initialized:
            self.agent_initialized = True
            print ("initializing agent...")
            self.agent.agent_init(use_gpu=args.gpu, agent_count=agent_count,
                                  rgb_image_count=self.rgb_image_count,
                                  depth_image_dim=self.depth_image_count * self.depth_image_dim,
                                  ir_dim=self.ir_count * ir_dim,
                                  ground_dim=self.ground_count * ground_dim,
                                  compass_dim=self.compass_count * compass_dim,
                                  target_dim=self.target_count * target_dim,
                                  model=args.model)
            self.reward_sum = np.zeros((agent_count), dtype=np.float32)
            dateinfo = datetime.datetime.now()
            self.logDirPath = args.log_file + dateinfo.strftime("%Y%m%d%H%M%S") + "/"
            os.makedirs(self.logDirPath)
            self.log_file = self.logDirPath + "reward.log"
            
            with open(self.log_file, 'w') as the_file:
                the_file.write('cycle, episode_reward_sum \n')
            
            self.agent.q_net.model.to_cpu()
            self.model_log = self.logDirPath + "model_" + str(self.agent.time - 1) + ".pkl"
            pickle.dump(self.agent.q_net.model, open(self.model_log, "wb"), -1)
            self.agent.q_net.model.to_gpu()
            self.agent.q_net.optimizer.setup(self.agent.q_net.model)
            
            action, q_now = self.agent.agent_start(observation, reward)
            self.send_action(action)

            self.q_log = self.logDirPath + "q.pkl"
            pickle.dump(q_now, open(self.q_log, "wb"), -1)
                        
        else:
            if args.mode_distribute:
                self.thread_event.wait()
                
            self.cycle_counter += 1
            self.reward_sum += reward

            if end_episode:
                self.agent.agent_end(reward)                
                with open(self.log_file, 'a') as the_file:
                    the_file.write(str(self.agent.time - 1) + ',' + str(self.reward_sum) + '\n')

                self.agent.q_net.model.to_cpu()
                self.model_log = self.logDirPath + "model_" + str(self.agent.time - 1) + ".pkl"
                pickle.dump(self.agent.q_net.model, open(self.model_log, "wb"), -1)
                self.agent.q_net.model.to_gpu()
                self.agent.q_net.optimizer.setup(self.agent.q_net.model)
                
                self.reward_sum = np.zeros((agent_count), dtype=np.float32)
                action = self.agent.agent_start(observation, reward)  # TODO
                self.send_action(action)
            else:
                action, eps, q_now = self.agent.agent_step(reward, observation)
                self.send_action(action)
                self.agent.agent_step_update(reward, action, eps)
        
                pickle.dump(q_now, open(self.q_log, "ab"), -1)

        if args.mode_distribute:
            self.thread_event.set()

cherrypy.config.update({'server.socket_host': args.ip,
                        'server.socket_port': args.port})
WebSocketPlugin(cherrypy.engine).subscribe()
cherrypy.tools.websocket = WebSocketTool()
cherrypy.config.update({'engine.autoreload.on': False})
config = {'/ws': {'tools.websocket.on': True,
                  'tools.websocket.handler_cls': AgentServer}}
cherrypy.quickstart(Root(), '/', config)

