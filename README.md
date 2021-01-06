# NLH Poker Reinforcement Learning agents
## Acknowledgement
This project is a course project of ESTR3108 in CUHK. It is based on [PyPokerEngine](https://github.com/ishikota/PyPokerEngine), detailed docs on env could be found at its [doc site](https://ishikota.github.io/PyPokerEngine/).
## Preliminaries
```sh
# Game environment
pip install PyPokerEngine
# Pytorch
pip install torch
# Process the result
pip install scipy
```
## Run
```sh
# Train AI agents
python train.py

# Test a model or just play a game
python test.py
```
## Agent list:
+ A2C Player
+ DQN Player
+ Q-Learning Agent
+ Card Player
+ Honest Player
+ Random Player
+ AllCall Player
+ Human Player

## Directory Structure
![Dir Tree](pic/dir-tree.png)

## Train Example
An example figure oczraining process.
![A2C Training](pic/a2c-vs-card.png)
## best configuration
+ LR=1e-4
+ gamma=0.95
+ Neural Network: 8->128->8, relu
+ Reward: No everage, /150


