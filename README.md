# When Mixnets Fail

This repository contains the artifact for the paper titled “When Mixnets Fail: Evaluating, Quantifying, and Mitigating the Impact of Adversarial Nodes in Mix Networks,” which has been accepted—subject to minor revision—at NDSS 2026.

## Initial setup and dependencies
You can execute the code on any standard laptop or workstation running Ubuntu 18.04 or higher. It is compatible with Python 3.8.10. Importantly, the artifact uses the same configuration as in the paper but with a reduced number of iterations, making it suitable for faster execution so that it can be readily run on a personal laptop or public infrastructure such as Google Colab.

Furthermore, the artifact has been optimized to run on systems with 16\,GB of RAM and 50\,GB of available disk space. These specifications allow users to reproduce results efficiently without requiring access to high-performance computing environments. We have also tested running the Git repository on Google Colab, which users may choose as an alternative.  

If you are interested in running the artifact on a laptop, please ensure that your system satisfies the following requirements: Ubuntu 18.04 or higher, Python 3.8.10, a minimum of 16\,GB of RAM, and at least 50\,GB of available disk space.  

All required dependencies for execution are listed in the `requirements.txt` file included in the repository and are summarized below:  

- matplotlib  
- numpy  
- pandas  
- pulp  
- scipy  
- simpy  
- tabulate  

However, to install all requirements automatically, you only need to run the following command once from the command line or within Google Colab before executing the project:  

`pip install -r requirements.txt`

## Hardware Requirements
The code has been tested on standard hardware with 16\,GB of RAM, 8 cores, and 50\,GB of available disk space. Alternatively, the artifact can be executed on Google Colab. To do so, you will need a Google account. Once signed in, you can clone the Git repository and run the code for free by following the instructions provided below.




# Project Structure

```text
.
mix_adversary
├── Algorithms.py                                 # This .py file contains a comprehensive set of functions for analyzing the paper.
├── Clients.py                                    # Simulates clients.
├── Main_Functions.py                             # DPMIX.py contains a comprehensive set of functions for analyzing DP-Mix.
├── main.py                                       # This file provides instructions regarding how to run the experiments described
│                                                 # in the main body of the paper.
├── Message_Genartion_and_mix_net_processing_.py  # This Python file, on behalf of clients, generates the messages to be sent
│                                                 # to the mixnet.
├── Message_.py                                   # Simulates the messages generated and sent by the clients.
├── Mix_Node_.py                                  # Simulates using discrete event simulation, a mixnode in mixnets.
├── NYM.py                                        # This .py file provides the main simulation components necessary
│                                                 # to simulate the NYM mixnet.
├── PLOTTER.py                                    # To plot the figures.
├── Routing.py                                    # This function helps to model the routing approaches.
│   Sim.py                                        # This .py file also includes the necessary simulation components                       
└──                                               # for reproducing simulations.                                 
 
```






# Evaluation Workflow

## Major Claims

- **(C1):** The first claim concerns the trend illustrated in Figure 3. Across all settings and scenarios, we observe that as the parameters vary—specifically, increasing the size of $S_h$ (Figure 3a), decreasing $K$ (Figure 3b), and increasing $\alpha$ (Figure 3c)—the values of $H_D$ and $\mathbb{P}(D)$ on the Y-axis consistently decrease, while $\mathbb{P}(R)$ increases. This claim is substantiated by Experiment E1, which generates Figure 3 and clearly demonstrates this trend.

- **(C2):** The second claim concerns the trend illustrated in Figure 4. Across all settings and scenarios, we observe that as the parameters vary—specifically, increasing the size of $S_h$ (Figure 4a), decreasing $K$ (Figure 4b), and increasing $\alpha$ (Figure 4c)—the values of DLM on the Y-axis consistently decrease. This claim is supported by Experiment E2, which produces Figure 4 and clearly demonstrates this trend.

- **(C3):** The third claim concerns the trend illustrated in Figure 5. Across all settings and scenarios, we observe that as the parameters vary—specifically, increasing the size of $S_h$ (Figure 5a), decreasing $K$ (Figure 5b), and increasing $\alpha$ (Figure 5c)—the values of CAM on the Y-axis consistently decrease. This claim is supported by Experiment E3, which produces Figure 5 and clearly demonstrates this trend.










## Experiments

## E1: [Reproducing Fig. 3; verifying Claim C1] [5 min]  

- To run this experiment, first execute the following command: `python3 main.py`
- Then enter `100` as the experiment ID when prompted.
  
- Upon completion, the results will be saved as Fig_3_a.png, Fig_3_b.png, and Fig_3_c.png in the `mix_adversary/Figures/` directory.
- Verification:

You may compare the generated figures with Figure 3 in the paper (shown below). Note that due to execution constraints (e.g., reduced number of iterations for personal machines or Google Colab), the reproduced figures may not match the paper exactly. For verification purposes, focus on the consistency of the observed trends—particularly whether the values increase or decrease as expected along the x-axis.

<img width="1085" height="257" alt="image" src="https://github.com/user-attachments/assets/8ca2e6e0-99a6-4c1b-881a-333a5bb12683" />




## E2: [Reproducing Fig. 4; verifying Claim C2] [40 min]  

- To run this experiment, first execute the following command: `python3 main.py`
- Then enter `200` as the experiment ID when prompted.
  
- Upon completion, the results will be saved as Fig_4_a.png, Fig_4_b.png, and Fig_4_c.png in the `mix_adversary/Figures/` directory.
- Verification:

You may compare the generated figures with Figure 4 in the paper (shown below). Note that due to execution constraints (e.g., reduced number of iterations for personal machines or Google Colab), the reproduced figures may not match the paper exactly. For verification purposes, focus on the consistency of the observed trends—particularly whether the values increase or decrease as expected along the x-axis.



<img width="1284" height="312" alt="image" src="https://github.com/user-attachments/assets/c5607033-a488-4170-9a75-60e596cab6ea" />



## E3: [Reproducing Fig. 5; verifying Claim C3] [15 min]  

- To run this experiment, first execute the following command: `python3 main.py`
- Then enter `300` as the experiment ID when prompted.
  
- Upon completion, the results will be saved as Fig_5_a.png, Fig_5_b.png, and Fig_5_c.png in the `mix_adversary/Figures/` directory.
- Verification:

You may compare the generated figures with Figure 5 in the paper (shown below). Note that due to execution constraints (e.g., reduced number of iterations for personal machines or Google Colab), the reproduced figures may not match the paper exactly. For verification purposes, focus on the consistency of the observed trends—particularly whether the values increase or decrease as expected along the x-axis.




<img width="1177" height="274" alt="image" src="https://github.com/user-attachments/assets/25ff6044-c22e-4eea-b64b-398f338662ff" />

## E*: [Others] [<1 h]  

- If you are interested in running additional experiments that lead to the generation of specific figures or tables (which are not necessarily part of the main claims), you should first execute:
  `python3 main.py`
- Then, use the table below to input the appropriate experiment ID corresponding to each figure or table.

- **Note**: Some of these figures were originally generated by running the experiments over extended periods (e.g., two weeks). As such, reproducing them fully may not be feasible on standard personal machines. However, even partial reproduction—such as generating one or two tables—should be sufficient to observe similar trends.



<img width="926" height="249" alt="image" src="https://github.com/user-attachments/assets/55d7663b-fc2d-4176-ab84-bb7b9b0cebd9" />




## Additional Notes

- This artifact has been created to reproduce the results presented in the paper “DP-Mix: Differentially Private Routing in Mix Networks”, accepted for publication at ACSAC 2025, and to provide an environment that makes the code usable for future research. You can execute the code on any standard laptop or workstation running Ubuntu 18.04 or higher. It is compatible with Python 3.8.10. Alternatively, the repository can be cloned and executed in a Google Colab environment.  

- Importantly, the artifact includes precisely the same configurations and settings used in the original DP-Mix evaluation. The only exception is that the number of iterations and some minor parameters have been scaled down to ensure feasibility on standard hardware. These configuration parameters can be reviewed and, where appropriate, modified in the initialization of `DP_Mix_Functions.py`. This adjustment ensures that the artifact remains practical for local execution, while still supporting future extensions where researchers may wish to adapt our approach to scenarios not explicitly tested in DP-Mix.  

- That said, modifying parameters requires a deeper understanding of mixnets. Mixnets are complex systems in which changes to a single parameter may influence others. In many cases, certain parameter combinations may be incompatible or may significantly affect the overall results. Additionally, several parameters are initialized based on prior work, and their default values were chosen to support meaningful comparisons. As such, arbitrary modifications may not yield valid outcomes.  

- However, the following parameters in the initialization of `DP_Mix_Functions.py` can be safely modified within specific intervals:  

1- `self.Iterations`: can be increased up to 30 to improve accuracy; note that this change increases computational cost exponentially.
  
2- `self.num_targets`: specifies the number of target messages in the simulations; can be set to any integer in [20,200].  

3- `self.run`: defines the duration of each simulation time slot; can be set to any real value in [0.05,1.0]. 
 
4- `self.delay1`: represents the average delay imposed on each message upon entering mixnodes; can be set to any real value in [0.01,0.08]. 
 

- Other parameters should not be modified, as they are tied to fixed design assumptions in mixnets. Altering them may lead to execution errors or invalid experimental results. If users wish to modify such parameters, we recommend contacting the authors directly for further guidance. 


 

- Execution notes: If the following warnings appear during execution, they can be safely ignored:  
```text
1) Gdk-CRITICAL **: 13:07:09.758: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed 

2) UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
```


