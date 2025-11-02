# -*- coding: utf-8 -*-
"""
Simulates clients
"""

#Client class to emulate a clietn with ID: cleints_is sending about 
from Message_ import message


class Client(object):
    
    def __init__(self,client_Id,n,num_targets,routing_paths,state,Target=True):
        self.ID = client_Id
        self.num_packets = n
        self.num_tar = num_targets
        self.paths = routing_paths
        self.T = Target
        self.state = state
        
        
        
    
    def message_generations(self):
        Messages = []
        
        for It in range(self.num_packets):

            M = message('It'+str(self.state+It+1),self.num_tar,self.ID,self.ID)

            M.path = self.paths[It]
            if (len(M.path))==1:
                print(self.paths)
            #print(M.path)
            if self.T:
                M.Loss = 0
            Messages.append(M)
            
        return Messages
