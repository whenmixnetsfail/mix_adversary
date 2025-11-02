# -*- coding: utf-8 -*-
"""
Main: This file provides instructions regarding how to run the experiments described in the main body of the paper.
"""


"""


...........................Experiments.....................................................
To run the experiments:
Enter 100 for E1

Enter 200 for E2

Enter 300 for E3

.............................Figures.......................................................

To run the experiments:
    
Enter 3 for Fig. 3

Enter 4 for Fig. 4

Enter 5 for Fig. 5

Enter 6 for Fig. 6

Enter 7 for Fig. 7 

Enter 891 for Fig. 8-a and Fig. 9-a

Enter 892 for Fig. 8-b and Fig. 9-b

Enter 893 for Fig. 8-c and Fig. 9-c


Enter 101 for Fig. 10-a
Enter 102 for Fig. 10-b
Enter 103 for Fig. 10-c

Enter 111 for Fig. 11-a
Enter 112 for Fig. 11-b
Enter 113 for Fig. 11-c

    
    
-------------------------------------Table-------------------------------------------------------------------------
Enter 1000 for Tab. 1

--------------------------------------------------------------------------------------------
"""

from Main_Functions import EXP_Mix

experiment_id = int(input("Please enter the ID of the experiment you wish to run: "))

Class_Mix = EXP_Mix(experiment_id)

