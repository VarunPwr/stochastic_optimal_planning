# Stochastic Optimal Planning
**Author : Varun Pawar <br>
Email : vpawar@ucsd.edu**

## Overview
Most modern robotics systems are stochastic in nature because of the uncertainty in sensor measurements. Here we solve the problem of planning optimal control when the system evolves stochastically in 3-dimensional space. More specifically, the project requires generating a safe trajectory for a differential drive robot. The robot must track a predetermined trajectory while avoiding collision with a few obstacles simultaneously.<br>
To complete the above objective, two different approaches are given. First is receding-horizon certainty equivalent control(CEC). CEC is a sub-optimal deterministic method and it is supposed to be solved using non-linear programming using Casadi framework. The other approach is policy iteration in a known model case
## Dependencies
python 3.7, matplotlib 3.4, and numpy 1.20. 

File strucutre			Description 
* **main.py :**			Implementation of receding horizon CEC and GPI algorithm. 
* **utils.py :**				Contains utility functions



## Results
![casidi_T1](https://user-images.githubusercontent.com/25801462/183261187-e2df68b0-0a9b-4e39-b615-f5fcd2a28190.gif)<br>
**Fig. 1: Motion plan followed by the agent using receding horizon CEC algorithm. T = 1**<br>
![casidi_T3](https://user-images.githubusercontent.com/25801462/183261290-b7469cb9-f706-49f2-8e07-46698214e999.gif)<br>
**Fig. 2: Motion plan followed by the agent using receding horizon CEC algorithm. T = 3**<br>
![casadi_T5](https://user-images.githubusercontent.com/25801462/183261298-7da80e8e-7494-427d-9ff6-e5d114fe551c.gif)<br>
**Fig. 3: Motion plan followed by the agent using receding horizon CEC algorithm. T = 5**<br>
![GPI_U6](https://user-images.githubusercontent.com/25801462/183261310-33e9a33b-dcb5-4bd9-b227-2e1540ef38fa.gif)<br>
**Fig. 4: Motion plan followed by the agent using GPI algorithm. $T=3$ and $|\mathcal{U}| = 6$**<br>
![GPI_U25](https://user-images.githubusercontent.com/25801462/183261312-b50b3a48-b97d-4acb-89be-d4c54cc704e1.gif)<br>
**Fig. 5: Motion plan followed by the agent using GPI algorithm. $T=3$ and $|\mathcal{U}| = 25$**<br>
![GPI_U50](https://user-images.githubusercontent.com/25801462/183261313-56123c4f-22e5-4599-845d-4b54203e591f.gif)<br>
**Fig. 6: Motion plan followed by the agent using GPI algorithm. $T=3$ and $|\mathcal{U}| = 50$**<br>
