# Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning

This is the the official implementation of Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning (SRDP) algorithm for the **IEEE RA-L 2024**, [<ins>"Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning"</ins>](https://ieeexplore.ieee.org/document/10423845). 

Offline Reinforcement Learning (RL) methods leverage previous experiences to learn better policies than the behavior policy used for data collection. However, they face challenges handling distribution shifts due to the lack of online interaction during training. To this end, we propose a novel method named State Reconstruction for Diffusion Policies (SRDP) that incorporates state reconstruction feature learning in the recent class of diffusion policies to address the problem of out-of-distribution (OOD) generalization. Our method promotes learning of generalizable state representation to alleviate the distribution shift caused by OOD states. To illustrate the OOD generalization and faster convergence of SRDP, we design a novel 2D Multimodal Contextual Bandit environment and realize it on a 6-DoF real-world UR10 robot, as well as in simulation, and compare its performance with prior algorithms. In particular, we show the importance of the proposed state reconstruction via ablation studies. In addition, we assess the performance of our model on standard continuous control benchmarks (D4RL), namely the navigation of an 8-DoF ant and forward locomotion of half-cheetah, hopper, and walker2d, achieving state-of-the-art results. Finally, we demonstrate that our method can achieve 167% improvement over the competing baseline on a sparse continuous control navigation task where various regions of the state space are removed from the offline RL dataset, including the region encapsulating the goal.
https://ieeexplore.ieee.org/document/10423845
## Requirements

This project requires [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) and includes code from the open-source [Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) project, which is licensed under the Apache License 2.0. Please see the ``environment_srdp.yaml`` for additional installation requirements.

## Train 
```.bash
python main.py --env_name [ENV_NAME] --device [0/1] --dp_type "ae" --lr_decay --seed [SEED]
```


## Citation
If you find our work useful, please cite:
```
@ARTICLE{10423845,
  author={Ada, Suzan Ece and Oztop, Erhan and Ugur, Emre},
  journal={IEEE Robotics and Automation Letters}, 
  title={Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning}, 
  year={2024},
  volume={9},
  number={4},
  pages={3116-3123},
  keywords={Behavioral sciences;Training;Task analysis;Adaptation models;Reinforcement learning;Uncertainty;Noise measurement;Reinforcement learning;deep learning methods;learning from demonstration},
  doi={10.1109/LRA.2024.3363530}}
```




