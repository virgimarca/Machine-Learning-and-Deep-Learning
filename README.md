# Machine-Learning-and-Deep-Learning
## Sim-to-Real transfer of Reinforcement Learning
This repository contains the code for the Machine Learning and Deep Learning course project.
In the folder classes our implementations can be found. In particular the following files implement the specified algorithms:
* vanilla.py: which implements the Vanilla Policy Gradient algorithm (VPG);
* actor_critic.py: where you can find Actor Critic Policy Gradient (ACPG);
* dropo.py: our implementation of the Domain Randomization Off Policy Optimization algorithm (DROPO);
* env/custom_hopper.py: an implemententation of the Hopper environment supporting Uniform Domain Randomization and a domain randomization distribution modelled as a truncated Gaussian used to test DROPO;

Inside the repository the following files are also present:
* train.py and test.py: the code used to respectively train and then test the VPG and the ACPG agents;
* ppo.py and trpo.py: which train and test Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO) respectively;
* dropo_data_collection.py and dropo_test.py: the code used to collect the dataset for DROPO and to later test said algorithm;

The folder UDR is also present and contains the hyperparameter tuning for the uniform domain randomization part and the enviroments with different parameters which where tested.

