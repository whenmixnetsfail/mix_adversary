# -*- coding: utf-8 -*-
"""
This Python file, on behalf of clients, generates the messages to be sent to the mixnet.
"""

import numpy as np
from scipy.stats import expon
from Message_ import message
from Clients import Client
from Algorithms import CirMixNet


class Message_Genartion_and_mix_net_processing(object):
    
    def __init__(self,env,Mixes,capacity,Fun_name,MNet,num_target,delay,W,L,nn,num_packets,param):
        
        self.env = env
        
        self.Mixes = Mixes
        
        self.capacity = capacity
        
        self.fun = Fun_name
        
        self.MNet = MNet
        
        self.NT = num_target
        
        self.delay = delay
        self.W = W
        self.L = L
        self.N = self.W*self.L
        self.nn = nn
        self.num_packets = num_packets
        self.R_Class = CirMixNet(self.NT,2,self.capacity,0.5,0.05,self.delay,2) 
        self.param = param

    def Prc(self):

        import math
        #This function is written to be used for generating the messages through the mix network

        ID = 0 #The id of the first target messages
        N_n = int(self.nn*self.W*(1/self.num_packets))
        Path_n = []
        for ii in range(N_n):
            
            if self.fun == "alpha":
                temp_path = self.R_Class.alpha_strategy([self.param],self.N,self.L,self.num_packets)[0]
                
            elif self.fun == "fix":
                temp_path = self.R_Class.fix_strategy(self.param,self.N,self.L,self.num_packets)  
                
            
            elif self.fun == "W":
                temp_path = self.R_Class.w_strategy(self.param,self.N,self.L,self.num_packets)                  
                
                
            Path_n.append(temp_path)
        
        
        
        
        for i in range(self.nn*self.W):#generate a fix number of messages to initiate the network
            TARGET = False
            target_id = -1
       
            client_id = 'Cl' 
            M = message('Message%02d' %i,self.NT,target_id,client_id)#message is being created
            M.path = Path_n[i//self.num_packets][i%self.num_packets]
            

            self.env.process(self.MNet.Message_Traveling(M))#message send to the mix net

        i = self.nn*self.W
        while True:# Create other messages by an exponential delay
            t2 = expon.rvs(scale=self.delay)#The exponential delay between two succeeding messages (each message includes n packets)
            yield self.env.timeout(t2)
            

            TARGET = False
            target_id = -1
            if(ID < self.NT):
                y = np.random.multinomial(1, [1/2,1/2], size=1)[0][0]
                if y==1:
                    TARGET = True
                if TARGET:
                    target_id = ID
                    ID = ID + 1
                    
            #self.num_packets = int(50*np.random.rand(1)[0])+1
            #client_id = 'Cl' 
            #M = message('Message%02d'%i,self.NT,target_id,client_id)
            if self.fun == "alpha":
                routing_paths = self.R_Class.alpha_strategy([self.param],self.N,self.L,self.num_packets)[0]
                
            elif self.fun == "fix":
                routing_paths = self.R_Class.fix_strategy(self.param,self.N,self.L,self.num_packets)  
                
            
            elif self.fun == "W":
                routing_paths = self.R_Class.w_strategy(self.param,self.N,self.L,self.num_packets)            
            
            
            Cl = Client(target_id,self.num_packets,self.NT,routing_paths,i,TARGET)
            MM_ = Cl.message_generations()
            #print(routing_paths)
            for M in MM_:
                self.env.process(self.MNet.Message_Traveling(M) ) 
                
            i = i + self.num_packets









