# NetworkTrafficScheduling (PhD Work Part1)
Network Traffic Scheduling using RL - First part of PhD Work

Dependencies: Ray.RLLib, Gym, Tensorflow

Approach in Learning
Deep Q Network (Training starts) -> Actor -Critic Policy Proximity Optimization (PPO)-> Optimal Results array -> Plot as gannt chart

Steps
1. Register the Environment with OpenAI Gym by going to the src folder (where setup.py is) and run command pip install -e . (pip need to be 23.1 or so)
2. Run: python main.py
3. To see the final solution python show_results.py 
4. To have the schedule plotted as gannt chart run python plotschedule.py

The details of the solution approach can be found at   
https://github.com/Abh4git/NetworkTrafficScheduling/blob/main/presentation/Part1-Traffic_Scheduling.pdf

Inspired and started based on Job Shop Scheduling done at the below link    
https://github.com/prosysscience/RL-Job-Shop-Scheduling

Our work is now a publication with Springer 
https://link.springer.com/chapter/10.1007/978-981-97-2004-0_13

