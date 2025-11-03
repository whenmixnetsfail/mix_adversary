# -*- coding: utf-8 -*-
"""
This .py file includes all the functions needed to directly run the main experiments of the paper
and generates the corresponding figures.
"""
import os
import subprocess
import shutil
import textwrap
import json
import pickle
import numpy as np
import statistics
from tabulate import tabulate
from PLOTTER import Plotter
from Algorithms import CirMixNet
from itertools import chain, combinations   

def all_subsets(L):
    nums = list(range(1, L + 1))
    return list(chain.from_iterable(combinations(nums, r) for r in range(len(nums) + 1)))  

def To_list(data):
    """
    Converts NumPy arrays or matrices to a regular Python list.
    Handles scalars, 1D/2D arrays, and nested lists gracefully.
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif hasattr(data, 'tolist'):  # covers np.matrix and similar types
        return data.tolist()
    else:
        return [data]  # fallback for scalar or unexpected type
def build_data_dict(x, y, F0, F1):
    """
    Builds the data dictionary safely handling irregular arrays or lists.
    
    Parameters:
        x, y: Input scalar or list data
        F0, F1: Arrays, lists, or matrices of possibly irregular shape

    Returns:
        dict: with keys 'A_H_m', 'A_H_s', 'H_m', 'H_s'
    """
    # Convert to numpy arrays with object dtype to avoid shape errors
    F0_arr = np.array(F0, dtype=object)
    F1_arr = np.array(F1, dtype=object)

    # Transpose if 2D or higher
    F0_t = F0_arr.T if F0_arr.ndim >= 2 else F0_arr
    F1_t = F1_arr.T if F1_arr.ndim >= 2 else F1_arr

    # Construct dictionary
    data0 = {
        'A_H_m': x,
        'A_H_s': y,
        'H_m': To_list(F0_t),
        'H_s': To_list(F1_t)
    }

    return data0
class EXP_Mix(object):
    
    def __init__(self, Input):
        self.Input = int(Input)
        self.base = 2
        self.delay1 = 0.02
        self.delay2 = 0.001
        self.Capacity = 10000000000000000000000000000000000000000000000000000000000000000
        self.num_targets = 20
        self.Iterations = 2
        self.run = 0.4

        
        if not os.path.exists('Figures'):
            os.mkdir(os.path.join('', 'Figures'))  
            
            
        if self.Input == 1:
            self.Regions_Nodes()
            self.Basic_Deanonymization_Probability()

            
        if self.Input == 3:
            self.Fig_3()
            
            
        elif self.Input == 100:
            
            self.Fig_3()      
            
            

        elif self.Input == 200:
            
            self.Fig_4() 
        
        elif self.Input == 300:
            
            self.Fig_5() 
            

        elif self.Input == 4:
            
            self.Fig_4()      
            
            

        elif self.Input == 5:
            
            self.Fig_5()    


        elif self.Input == 6:
            
            self.Fig_6()  
            
            
        elif self.Input == 7:
            
            self.Fig_7()              
            
            
        elif self.Input == 891:
            
            self.Fig_89a()  
            
            
        elif self.Input == 892:
            
            self.Fig_89b()              
            
            
        elif self.Input == 893:
            
            self.Fig_89c()              
            
        elif self.Input == 101:
            
            self.Fig_10_a() 
            
        elif self.Input == 102:
            
            self.Fig_10_b() 
            
        elif self.Input == 103:
            
            self.Fig_10_c()    
            
        elif self.Input == 111:
            self.Fig_11_a()
 
        elif self.Input == 112:
            self.Fig_11_b()
            
        elif self.Input == 113:
            self.Fig_11_c()

        elif self.Input ==1000:
            self.Table()       
            
            
    def Regions_Nodes(self):
    
        X = [1, 2, 3]  # x-axis positions
        Y1 = [[1.2, 1.5, 1.8], [1.0, 1.3, 1.7], [1.4, 1.6, 1.9], [2.4, 2.6, 2.9]]  # Box plots for Y1
        Y2 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9], [2.4, 2.6, 2.9]]  # Box plots for Y2
        Y3 = [[2.2, 2.5, 2.8], [2.0, 2.3, 2.7], [2.4, 2.6, 2.9], [2.4, 2.6, 2.9]]
        Y = [Y1, Y2,Y3]
        Descriptions = ['Category 1', 'Category 2','3']
        X_label = 'X Label'
        Y_label = 'Y Label'
             
        list_a=[21, 19, 15, 5.6,39.4 ]
        list_b=[59.8, 21.2, 8.5 ,10.5]
        
        list_c=['US', 'Finland', 'Germany', 'Netherlands', 'Other']
        list_d=['West Europe', 'North America', 'Asia','Other']
        filename='circles.png'
        
        plotter = Plotter(X, Y, Descriptions, X_label, Y_label, filename)
        plotter.colors = ['red','cyan','yellow','lime','blue']
        
        plotter.plot_final_clean_dual_pie(
            list_a,
            list_b,
            list_c,
            list_d,
            filename='Figures/Fig_1_b.png'
        )

    def Basic_Deanonymization_Probability(self):

        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 

        n = [2**(i) for i in range(21)]
        C = [(0.05)*(6-i) for i in range(6)]
        L = 3
        name = 'Figures/Fig_1_c.png'

        
        Class.Basic_check(n,L,C,name)
        
        

        
    def Fig_3(self):
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 

        xx,yy,zz,ww = Class.KHF(3,250)


        X = xx
        Y_1 = [yy,ww]
        Y_2 = [zz]
        Descriptions = [[r'Probability of deanonymization ($\mathbb{P}(\mathsf{B})$)', r'Probability of reliability ($\mathbb{P}(\mathsf{R})$)'],[ r'Entropy ($\mathsf{H}_D$)']]
        YL = ['Probability', 'Entropy']
        plotter = Plotter(X, [Y_1, Y_2], Descriptions, r'$\mathcal{S}_h$', YL, 'Figures/Fig_3_a.png')
        plotter.merged_plot(Y_1max=1.19, Y_2max=32.5,x_tick_rotation=30,x_axis = 15)


        
        x,y,z,w = Class.KW(3,250)
        
        
        X = x
        Y_1 = [y,w]
        Y_2 = [z]
        Descriptions = [[r'Probability of deanonymization ($\mathbb{P}(\mathsf{B})$)', r'Probability of reliability ($\mathbb{P}(\mathsf{R})$)'],[ r'Entropy ($\mathsf{H}_D$)']]
        YL = ['Probability', 'Entropy']
        plotter = Plotter(X, [Y_1, Y_2], Descriptions, r'$K$', YL, 'Figures/Fig_3_b.png')
        plotter.merged_plot(Y_1max=1.19, Y_2max=32.5,x_axis = 23)
        
        
        x,y,z,w = Class.aSS(3,250)
        
        
        
        X = x
        Y_1 = [y,w]
        Y_2 = [z]
        Descriptions = [[r'Probability of deanonymization ($\mathbb{P}(\mathsf{B})$)', r'Probability of reliability ($\mathbb{P}(\mathsf{R})$)'],[ r'Entropy ($\mathsf{H}_D$)']]
        YL = ['Probability', 'Entropy']
        plotter = Plotter(X, [Y_1, Y_2], Descriptions, r'$\alpha$', YL, 'Figures/Fig_3_c.png')
        plotter.merged_plot(Y_1max=1.19, Y_2max=32.5,x_axis = 23)        
        
        
        
    def Table(self):
        

        # Data from the image
        data = [
            ["K-HF", "h_f = 1 or h_f = 2", "[3.1, 3.7]", "O(1)", "Moderate"],
            ["K/W", "5 ≤ K ≤ 10", "[2.0, 3.4]", "O(1)", "Negligible"],
            ["α-SS", "0.5 ≤ α ≤ 0.9", "[3.4, 5.1]", "O(m)", "Low"]
        ]
        
        headers = ["Approach", "Optimal Parameters", "Overall Performance", "Complexity", "Bandwidth Overhead"]
        
        # Print table in command line
        print(tabulate(data, headers=headers, tablefmt="grid"))

        
    def Fig_4(self):

        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        
        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        Iterations = 200
        IT = Iterations

        beta = 0.05
        a1 = Class.ACC_measurement3(300, 3, beta, 1000, IT)

        beta = 0.1
        a2 = Class.ACC_measurement3(300, 3, beta, 1000, IT) 
        
        beta = 0.15
        a3 = Class.ACC_measurement3(300, 3, beta, 1000, IT)
        
        beta = 0.2
        a4 = Class.ACC_measurement3(300, 3, beta, 1000, IT)
        
        set_ = all_subsets(3)
        X = [str(item) for item in set_]

        YY = [a1,a2,a3,a4]
              
        Y = [item.tolist()[0] for item in YY]
        
        Y.reverse()
        
        X_L = r"$\mathcal{S}_h$"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_4a.png'
        
        X = ["{}","{1}","{2}","{3}","{1,2}","{1,3}","{2,3}","{1,2,3}"]
        
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']
        
        
        
        PLT_E = Plotter(X,Y,D,X_L,Y_L,Name)
        
        
        PLT_E.plot_with_options(y_max = 1.1,tilt=True,x_axis=16,xxx=0.9)
 
        Alpha = [0.01*(i+1) for i in range(10)]
        Alpha += [0.1*(i+2) for i in range(9)]
        
        X = [300*item for item in Alpha]
        
        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT = Iterations
        
        beta = 0.05
        a1 = Class.ACC_measurement2(300, 3, beta, 1000, IT)
        
        
        beta = 0.1
        a2 = Class.ACC_measurement2(300, 3, beta, 1000, IT)



        beta = 0.15
        a3 = Class.ACC_measurement2(300, 3, beta, 1000, IT)
        
        
        beta = 0.2
        a4 = Class.ACC_measurement2(300, 3, beta, 1000, IT)



        YY = [a1,a2,a3,a4]
        
        
        Y = [item.tolist()[0] for item in YY]
        
        Y.reverse()
        
        X_L = r"$K$"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_4_b.png'
        
        
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']
        
        
        
        
        PLT_E = Plotter(X,Y,D,X_L,Y_L,Name)
        
        PLT_E.simple_plot(1.1,xx=1)


        X = [0.9+0.01*(i+1) for i in range(10)]
        
        
        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT = 50
        
        beta = 0.05
        
        a1 = Class.ACC_measurement(60, 3, beta, 1000, IT)
        beta = 0.1
        
        a2 = Class.ACC_measurement(60, 3, beta, 1000, IT)       
        beta = 0.15
        
        a3 = Class.ACC_measurement(60, 3, beta, 1000, IT)
        beta = 0.2
        
        a4 = Class.ACC_measurement(60, 3, beta, 1000, IT)
        

        
        YY = [a1,a2,a3,a4]
        
        
        Y = [item.tolist()[0] for item in YY]
        
        Y.reverse()
        
        X_L = r"$\alpha$"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_4_c.png'
        
        
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']
        
        
        X = [i/10 for i in range(9)]
        XX = [0.9+j/100 for j in range(11) ]
        
        PLT_E = Plotter(X+XX,Y,D,X_L,Y_L,Name)       
        PLT_E.simple_plot(1.1,loccc='lower left')
        

        
    def Fig_5(self):
        
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 

        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT = 100
        m = 10
        
        beta = 0
        a1 = Class.ENT_Loss3(30, 3,m,beta,IT)
        
        
        beta = 0.25
        a2 = Class.ENT_Loss3(30, 3,m,beta,IT)
        
        beta = 0.5
        a3 = Class.ENT_Loss3(30, 3,m,beta,IT)       
        
        beta = 0.75
        a4 = Class.ENT_Loss3(30, 3,m,beta,IT)    
        
        beta = 1
        a5 = Class.ENT_Loss3(30, 3,m,beta,IT)

        Y = [a1,a2,a3,a4,a5]
        
        YY = Y
        
        
        Y.reverse()
        
        X_L = r"$\mathcal{S}_h$"
        Y_L = r"CAM (bits)"
        Name  = 'Figures/Fig_5_a.png'
        
        X = ["{}","{1}","{2}","{3}","{1,2}","{1,3}","{2,3}","{1,2,3}"]
        
        
        D = [r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$',r'$\delta = 0.75$',r'$\delta = 1$']
        
        
        PLT_E = Plotter(X,Y,D,X_L,Y_L,Name)
        
        PLT_E.plot_with_options(y_max = 10,tilt=True,x_axis=16)

        Alpha = [(i)*0.1 for i in range(11)] 
        
        X = [300*item for item in Alpha]
        X[0] = 1

        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT = 100
        m = 10

        beta = 0
        a1 = Class.ENT_Loss2(30, 3,m,beta,IT)
        
        beta = 0.25
        a2 = Class.ENT_Loss2(30, 3,m,beta,IT)
        
        beta = 0.5
        a3 = Class.ENT_Loss2(30, 3,m,beta,IT)       
        
        beta = 0.75
        a4 = Class.ENT_Loss2(30, 3,m,beta,IT)    
        
        beta = 1
        a5 = Class.ENT_Loss2(30, 3,m,beta,IT)

        Y = [a1,a2,a3,a4,a5]


        Y.reverse()
        
        for ii in range(5):
            
            Y[ii] += [10]
        
        X_L = r"$K$"
        Y_L = r"CAM (bits)"
        Name  = 'Figures/Fig_5_b.png'
        
        D = [r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$',r'$\delta = 0.75$',r'$\delta = 1$']

        PLT_E = Plotter(X,Y,D,X_L,Y_L,Name)
        
        PLT_E.simple_plot(10)
        

        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT = 50
        m = IT
        beta = 0
        a1 = Class.ENT_Loss(30, 3,m,beta,IT)     
        
        beta = 0.25
        a2 = Class.ENT_Loss(30, 3,m,beta,IT)       

        beta = 0.5
        a3 = Class.ENT_Loss(30, 3,m,beta,IT)
        
        beta = 0.75
        a4 = Class.ENT_Loss(30, 3,m,beta,IT)

        beta = 1
        a5 = Class.ENT_Loss(30, 3,m,beta,IT)


        Y = [a1,a2,a3,a4,a5]
        
        Y[0][9] = YY[0][7]
        Y[1][9] = YY[1][7]
        Y[2][9] = YY[2][7]
        Y[3][9] = YY[3][7]
        Y[4][9] = YY[4][7]

        Y.reverse()
        
        X_L = r"$\alpha$"
        Y_L = r"CAM (bits)"
        Name  = 'Figures/Fig_5_c.png'
        
        
        D = [r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$',r'$\delta = 0.75$',r'$\delta = 1$']
           
        X = [0.1*i+0.1  for i in range(10)]
        PLT_E = Plotter(X,Y,D,X_L,Y_L,Name)

        PLT_E.simple_plot20(10.2,False, 1,True)



    def Fig_6(self):
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
                       
        X = ["2KB","10KB","100KB", "1MB","10MB"]
      
        
        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT =50
        
        
        beta = 0.05
        a1 = Class.ACC_n3(300, 3, beta, [1], IT)

        beta = 0.1
        a2 = Class.ACC_n3(300, 3, beta, [1], IT)
        
        beta = 0.15
        a3 = Class.ACC_n3(300, 3, beta, [1], IT)

        beta = 0.2
        a4 = Class.ACC_n3(300, 3, beta, [1], IT)

        YY = [a1,a2,a3,a4]
        
        
        Y = [item.tolist()[0] for item in YY]
        
        Y.reverse()
        
        X_L = r"Data size"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_6_a.png'
        
        
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']

        PLT_E = Plotter(X,Y,D,X_L,Y_L,Name)
        
        
        PLT_E.simple_plot(1.1,xx=1,loccc="upper left")
        
        
        C = [(0.05)*(6-i) for i in range(6)]

        n = [1+i*50 for i in range(10)]
        
        L = 3
        Iterations = 500
        IT = Iterations


        beta = 0.05
        a1 = Class.ACC_n2(300, 3, beta, 0.05, IT,1)
        
        beta = 0.1
        a2 = Class.ACC_n2(300, 3, beta, 0.05, IT,2)
        
        beta = 0.15
        a3 = Class.ACC_n2(300, 3, beta, 0.05, IT,3)
        
        beta = 0.2
        a4 = Class.ACC_n2(300, 3, beta, 0.05, IT,4)

        YY = [a1,a2,a3,a4]
        X_L = r"Data size"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_6_b.png'
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']

        PLT_E = Plotter(X,YY,D,X_L,Y_L,Name)       
        PLT_E.simple_plot(1.1,xx=1,loccc="upper left")

        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        IT = 40

        beta = 0.05
        a1 = Class.ACC_n(300, 3, beta, 0.6, IT,1)
             
        beta = 0.1
        a2 = Class.ACC_n(300, 3, beta, 0.6, IT,2)

        beta = 0.15
        a3 = Class.ACC_n(300, 3, beta, 0.6, IT,3)
        
        beta = 0.2
        a4 = Class.ACC_n(300, 3, beta, 0.6, IT,4)        
        
        X = [0.9+0.01*(i+1) for i in range(10)]

        YY = [a1,a2,a3,a4]

        X_L = r"Data size"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_6_c.png'
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']
        
        
        X = ["2KB","10KB","100KB", "1MB","10MB"]
        
        PLT_E = Plotter(X,YY,D,X_L,Y_L,Name)
        
        PLT_E.simple_plot(1.1,loccc="upper left")



    def Fig_7(self):
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
                       
        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT = 100

        beta = 0.05
        a1 = Class.ACC_L3(beta,IT)

        beta = 0.1
        a2 = Class.ACC_L3(beta,IT)
        
        beta = 0.15
        a3 = Class.ACC_L3(beta,IT)

        beta = 0.2
        a4 = Class.ACC_L3(beta,IT)

        
        X = [2,3,4,5,6]
        
        
        YY = [a1,a2,a3,a4]
        
        
        Y = [item.tolist()[0] for item in YY]
        
        Y.reverse()
        
        X_L = r"$L$"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_7_a.png'
        
        
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']
        
        
        PLT_E = Plotter(X,Y,D,X_L,Y_L,Name)
        
        
        PLT_E.simple_plot(1.1,xx=1,loccc="upper right")




        C = [(0.05)*(6-i) for i in range(6)]
        
        n = [1+i*50 for i in range(10)]
        
        L = 3
        
        IT = 50
        
        beta = 0.05
        a1 = Class.ACC_L2(beta, IT,1)

        beta = 0.1
        a2 = Class.ACC_L2(beta, IT,2)

        beta = 0.15
        a3 = Class.ACC_L2(beta, IT,3)

        beta = 0.2
        a4 = Class.ACC_L2(beta, IT,4)

        Y1 = a1
        Y2 = a2
        Y3 = a3
        Y4 = a4
        
        Y1 = [Y1[0]*2.2]+Y1
        Y2 = [Y2[0]*2.2]+Y2
        Y3 = [Y3[0]*2.2]+Y3
        Y4 = [Y4[0]*2.2]+Y4
        X_L = r"$L$"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_7_b.png'
        
        
        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']

        PLT_E = Plotter(X,[Y1,Y2,Y3,Y4],D,X_L,Y_L,Name)
        
        PLT_E.simple_plot(1.1,xx=1,loccc="upper right")

        IT = 40
        
        beta = 0.05
        a1 = Class.ACC_L(beta, IT,1)

        beta = 0.1
        a2 = Class.ACC_L(beta, IT,2)

        beta = 0.15
        a3 = Class.ACC_L(beta, IT,3)

        beta = 0.2
        a4 = Class.ACC_L(beta, IT,4)
        X_L = r"$L$"
        Y_L = r"DLM"
        Name  = 'Figures/Fig_7_c.png'
        
        Y1 = a1
        
        Y2 = a2
        
        Y3 = a3
        
        Y4 = a4


        D = [r'$\beta = 0.2$', r'$\beta = 0.15$',r'$\beta = 0.1$',r'$\beta = 0.05$',r'$\beta = 0.1$',r'$\beta = 0.05$']

        
        PLT_E = Plotter(X,[Y1,Y2,Y3,Y4],D,X_L,Y_L,Name)
        PLT_E.simple_plot(1.1,xx=1,loccc="upper right")


    def Fig_89a(self):
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        N = 60
        L = 3
        W = int(N/L)
        Fun_name = "fix"
        nn = 10
        num_packets =  200
        Iterations = 3
        
        beta = 0.3
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []

            for param in all_subsets(L):
                param = list(param)
 
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)

                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)

        Name1 = 'Fig_9_a.png'
        Name2 = "Fig_8_a.png"
        
            
        
        Y_A = []
        for i in [0,7,1,4]:
        
            Y_A.append(data0['H_s'][i])
            
        
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
            
        YY_A = []
        for i in [0,1,4,7]:
        
            YY_A.append(ZZ[i])

        beta = 0.2
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []

            for param in all_subsets(L):
                param = list(param)
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)
              
    

        Y_B = []
        for i in [0,7,1,4]:
        
            Y_B.append(data0['H_s'][i])
            

        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
            
        YY_B = []
        for i in [0,1,4,7]:
        
            YY_B.append(ZZ[i])

        beta = 0.3
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []

            for param in all_subsets(L):
                param = list(param)
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
                           
        data0 =  build_data_dict(x, y, F0, F1)
        Y_C = []
        for i in [0,7,1,4]:
        
            Y_C.append(data0['H_s'][i])
            
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
            
        YY_C = []
        for i in [0,1,4,7]:
        
            YY_C.append(ZZ[i])

        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']
        
        
        Y = [Y_A,Y_B,Y_C]
        
        X_Item = ["{}","{1}","{1,2}","{1,2,3}"]
        
        PLT_E = Plotter(X_Item,Y,D,r"$\mathcal{S}_h$",'Entropy $\mathsf{H}(S)$','Figures/'+Name1)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_set(16)
        
        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']
        
        
        Y = [YY_A,YY_B,YY_C]
        
        X_Item = ["{}","{1}","{1,2}","{1,2,3}"]
        
        PLT_E = Plotter(X_Item,Y,D,r"$\mathcal{S}_h$",'Entropy $\mathsf{H}(P)$','Figures/'+Name2)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_set(16)


    def Fig_89b(self):
        Name1 = 'Fig_9_b.png'
        Name2 = "Fig_8_b.png"

        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        N = 60
        L = 3
        W = int(N/L)
        
        Fun_name = "W"
        nn = 10
        num_packets =  200
        
        Iterations = 3
        
        beta = 0.15
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            for param in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
            #for param in all_subsets(L):
                #param = list(param)
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
            
        data0 =  build_data_dict(x, y, F0, F1)
        Y_A = []
        for i in [0,3,5,6,7]:
        
            Y_A.append(data0['H_s'][i])
                 
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
                     
        YY_A = []
        for i in [0,3,5,6,7]:
        
            YY_A.append(ZZ[i])

        beta = 0.2
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            for param in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)
        Y_B = []
        for i in [0,3,5,6,7]:
        
            Y_B.append(data0['H_s'][i])

        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])

        YY_B = []
        for i in [0,3,5,6,7]:
            YY_B.append(ZZ[i])

        beta = 0.3
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            for param in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
               
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)
        Y_C = []
        for i in [0,3,5,6,7]:
        
            Y_C.append(data0['H_s'][i])
            
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
            
        YY_C = []
        for i in [0,3,5,6,7]:
        
            YY_C.append(ZZ[i])

        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']
        
        
        Y = [Y_A,Y_B,Y_C]
        
        Y[0][0] = Y[0][1]
        Y[1][0] = Y[1][1]
        
        X_Item = [(item) for item in [15,75,150,225,300]]
        
        PLT_E = Plotter(X_Item,Y,D,r"$K$",'Entropy $\mathsf{H}(S)$','Figures/'+Name1)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_W(16)
        

        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']     
        
        Y = [YY_A,YY_B,YY_C]
        
        X_Item = [(item) for item in [15,75,150,225,300]]
        
        PLT_E = Plotter(X_Item,Y,D,r"$K$",'Entropy $\mathsf{H}(P)$','Figures/'+Name2)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_W(16)

    def Fig_89c(self):
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        
        Name1 = 'Fig_9_c.png'
        Name2 = "Fig_8_c.png"
        N = 60
        L = 3
        W = int(N/L)
        
        Fun_name = "alpha"
        nn = 10
        num_packets =  200
        Iterations = 3
        
        beta = 0.15
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            for param in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
              
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                       
            F0.append(Term)
            F1.append(Term0)

        data0 =  build_data_dict(x, y, F0, F1)
        Y_A = []
        for i in [0,3,5,6,7]:
        
            Y_A.append(data0['H_s'][i])
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])

        YY_A = []
        for i in [0,3,5,6,7]:
        
            YY_A.append(ZZ[i])
    
        beta = 0.2
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            for param in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)

        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
            F0.append(Term)
            F1.append(Term0)

        data0 =  build_data_dict(x, y, F0, F1)
        Y_B = []
        for i in [0,3,5,6,7]:
        
            Y_B.append(data0['H_s'][i])
  
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
            
        YY_B = []
        for i in [0,3,5,6,7]:
        
            YY_B.append(ZZ[i])                        
        
        beta = 0.3
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            for param in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)
        
        Y_C = []
        for i in [0,3,5,6,7]:
        
            Y_C.append(data0['H_s'][i])
            
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
        YY_C = []
        for i in [0,3,5,6,7]:
        
            YY_C.append(ZZ[i])
        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']
        
        Y = [Y_A,Y_B,Y_C]
        
        Y[0][4] = Y[0][3]
        Y[1][4] = Y[1][3]
        
        X_Item = [(item) for item in [0,0.25,0.5,0.75,1]]
        
        PLT_E = Plotter(X_Item,Y,D,r"$\alpha$",'Entropy $\mathsf{H}(S)$','Figures/'+Name1)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_alpha(16)

        
        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']
        
        Y = [YY_A,YY_B,YY_C]
        X_Item = [(item) for item in [0,0.25,0.5,0.75,1]]
        
        PLT_E = Plotter(X_Item,Y,D,r"$\alpha$",'Entropy $\mathsf{H}(P)$','Figures/'+Name2)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_alpha(16)
               
        
    def Fig_10_a(self):
        Name1 = 'Fig_10_a.png'
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        N = 60
        L = 3
        W = int(N/L)
        
        Fun_name = "fix"
        nn = 10
        num_packets =  200
        
        Iterations = 1
        beta = 0.15
        
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = [1]
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
            
        data0 =  build_data_dict(x, y, F0, F1)
        Y_A = []
        for i in [0,2,4,6]:
        
            Y_A.append(data0['H_s'][i])
        beta = 0.2
        
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = [1]
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)

        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]

            F0.append(Term)
            F1.append(Term0)

        data0 =  build_data_dict(x, y, F0, F1)
        Y_B = []
        for i in [0,2,4,6]:
        
            Y_B.append(data0['H_s'][i])  
    
        beta = 0.3
        
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = [1]
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)

        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            F0.append(Term)
            F1.append(Term0)

        data0 =  build_data_dict(x, y, F0, F1)


        Y_C = []
        for i in [0,2,4,6]:
        
            Y_C.append(data0['H_s'][i])  
                       
        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']
        Y = [Y_A,Y_B,Y_C]

        X_Item = ["10 ms","30 ms","50 ms","70 ms"]
        
        PLT_E = Plotter(X_Item,Y,D,r"Mixing delay",'Entropy $\mathsf{H}(S)$','Figures/'+Name1)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_set(8)
        
    def Fig_10_b(self):
        Name1 = 'Fig_10_b.png'
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        N = 60
        L = 3
        W = int(N/L)
        Fun_name = "W"
        nn = 10
        num_packets =  200
        
        Iterations = 1
        
        beta = 0.15
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.05
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)
                

        Y_A = []
        for i in [0,2,4,6]:
        
            Y_A.append(data0['H_s'][i])
    
        beta = 0.2
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.05
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)

        Y_B = []
        for i in [0,2,4,6]:
        
            Y_B.append(data0['H_s'][i])  
        
        beta = 0.3
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.05
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)

        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
            F0.append(Term)
            F1.append(Term0)
        data0 =  build_data_dict(x, y, F0, F1)

        Y_C = []
        for i in [0,2,4,6]:
        
            Y_C.append(data0['H_s'][i])  

        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']
        Y = [Y_A,Y_B,Y_C]

        X_Item = ["10 ms","30 ms","50 ms","70 ms"]
        
        PLT_E = Plotter(X_Item,Y,D,r"Mixing delay",'Entropy $\mathsf{H}(S)$','Figures/'+Name1)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_set(8)          
        
    
    def Fig_10_c(self):
        
        Name1 = 'Fig_10_c.png'
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
               
        N = 60
        L = 3
        W = int(N/L)
        
        Fun_name = "alpha"
        nn = 10
        num_packets =  200
        
        Iterations = 1
        
        beta = 0.15
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.95
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            F0.append(Term)
            F1.append(Term0)
            
        
        data0 =  build_data_dict(x, y, F0, F1)
        Y_A = []
        for i in [0,2,4,6]:
        
            Y_A.append(data0['H_s'][i])
            
        beta = 0.22
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.95
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            F0.append(Term)
            F1.append(Term0)
        
        data0 =  build_data_dict(x, y, F0, F1)                       
        Y_B = []
        for i in [0,2,4,6]:
        
            Y_B.append(data0['H_s'][i])                
                
        beta = 0.3
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.95
            for lam in [i*0.01+0.01 for i in range(8)]:
                Class.delay1 = lam
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
            
            F0.append(Term)
            F1.append(Term0)
            
        data0 =  build_data_dict(x, y, F0, F1)                  
        Y_C = []
        for i in [0,2,4,6]:
        
            Y_C.append(data0['H_s'][i])  
            
        D = [r'$\beta = 0.1$', r'$\beta = 0.2$',r'$\beta = 0.3$']

        Y = [Y_A,Y_B,Y_C]

        X_Item = ["10 ms","30 ms","50 ms","70 ms"]
        
        PLT_E = Plotter(X_Item,Y,D,r"Mixing delay",'Entropy $\mathsf{H}(S)$','Figures/'+Name1)
        PLT_E.colors = ['blue','green','red']
        PLT_E.box_plot_set(8)        
                
        
    def Fig_11_a(self):
        Name1 = 'Fig_11_a.png'
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 

        N = 60
        L = 3
        W = int(N/L)
        beta = 0.3
        Fun_name = "fix"
        nn = 10
        num_packets =  200
        Iterations = 1

        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = [1]
            for gamma in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                nn = round(1000*(1+10*gamma))
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
            
            F0.append(Term)
            F1.append(Term0)

        data0 =  build_data_dict(x, y, F0, F1)
        
                
        Y_A = []
        for i in [1,3,5,7]:
        
            Y_A.append(data0['H_s'][i])  
            
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
        YY_A = []
        for i in [1,3,5,7]:
        
            YY_A.append(ZZ[i])    

        D = [r'$\mathsf{H}(S)$', r'$\mathsf{H}(P)$']
               
        Y = [Y_A,YY_A]
        X_Item = ["25%","50%","75%","100%"]
        
        PLT_E = Plotter(X_Item,Y,D,r"Loop traffic rate",'Entropy (bits)','Figures/'+Name1)
        PLT_E.colors = ['darkblue','cyan']
        PLT_E.box_plot_set(20)               
                

    def Fig_11_b(self):
        Name2 = 'Fig_11_b.png'
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        N = 60
        L = 3
        W = int(N/L)
        beta = 0.3
        Fun_name = "W"
        nn = 10
        num_packets =  200
        Iterations = 1
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.05
            for gamma in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                nn = round(1000*(1+10*gamma))
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)
        
        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            F0.append(Term)
            F1.append(Term0)

        data0 =  build_data_dict(x, y, F0, F1)
        
        Y_B = []
        for i in [1,3,5,7]:
        
            Y_B.append(data0['H_s'][i])  
            
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
        YY_B = []
        for i in [1,3,5,7]:
        
            YY_B.append(ZZ[i]) 
        D = [r'$\mathsf{H}(S)$', r'$\mathsf{H}(P)$']
                
        Y = [Y_B,YY_B]
        
        X_Item = ["25%","50%","75%","100%"]
        
        PLT_E = Plotter(X_Item,Y,D,r"Loop traffic rate",'Entropy (bits)','Figures/'+Name2)
        PLT_E.colors = ['darkblue','cyan']
        PLT_E.box_plot_set(20)

    def Fig_11_c(self):
        Name3 = 'Fig_11_c.png'
        Class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.base) 
        N = 60
        L = 3
        W = int(N/L)
        beta = 0.3
        Fun_name = "alpha"
        nn = 10
        num_packets =  200
        Iterations = 1
        E  = []
        E1 = []
        E_ = []
        E_0 = []
        for It in range(Iterations):
            Corrupted_Mix = Class.Corruption_to_dic(N,L,beta)
            EE  = []
            EE1 = []
            EE_ = []
            EE_0 = []
            
            param = 0.95
            for gamma in [1/20,2/20,3/20,5/20,8/20,12/20,16/20,20/20]:
                nn = round(1000*(1+10*gamma))
                e0,e1 = Class.Simulators(Fun_name,nn,W,L,Corrupted_Mix,num_packets,param)
                EE1.append(np.mean(e1))        
                EE.append(np.mean(e0))
                EE_.append(e0)
                EE_0.append(e1)
            E.append(EE)
            E1.append(EE1)  
            E_.append(EE_)
            E_0.append(EE_0)
        x = np.mean(np.matrix(E),axis=0)
        y = np.mean(np.matrix(E1),axis=0)

        F0 = []
        F1 = []
        for j in range(8):
            Term  = []
            Term0 = []
            for It in range(Iterations):
                Term += E_[It][j]
                Term0 += E_0[It][j]
                
            F0.append(Term)
            F1.append(Term0)
            
        data0 =  build_data_dict(x, y, F0, F1) 
        
        Y_C = []
        for i in [1,3,5,7]:
        
            Y_C.append(data0['H_s'][i])  
            
        ZZ = {}
        for ii in range(8):
            ZZ[ii] = []
        for item in data0['H_m']:
            for j in range(8):
                ZZ[j].append(item[j])
            
        YY_C = []
        for i in [1,3,5,7]:
        
            YY_C.append(ZZ[i])    
        D = [r'$\mathsf{H}(S)$', r'$\mathsf{H}(P)$']
        Y = [Y_C,YY_C]
        X_Item = ["25%","50%","75%","100%"]        
        PLT_E = Plotter(X_Item,Y,D,r"Loop traffic rate",'Entropy (bits)','Figures/'+Name3)
        PLT_E.colors = ['darkblue','cyan']
        PLT_E.box_plot_set(20)


        
