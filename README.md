# College-Admissions-Baseline

Inspired by the gym environments from [POCAR](https://github.com/ericyangyu/pocar)

A [Gym](https://github.com/openai/gym) environment that simulates the process of college admissions, particularly exploring the relationship between economic burden and acceptance. This environment provides fairness and utility metrics and generates plots to assess the performance of different RL-based decision-making algorithms in a simulation, allowing concrete baseline assessment of their long-term fairness.

Install [Anaconda](https://docs.anaconda.com/anaconda/install/) to setup a virtual environment.

Run the following commands
```
conda create -n college python=3.8
conda activate college
pip install -r requirements.txt
```

Usage:
Navigate to the directory where `main.py` is located and then run the following:

```
python main.py
```

This will train two agents - [A-PPO and G-PPO](https://github.com/ericyangyu/pocar) and evaluate their fairness and utility performance in our environment, outputting graphs.

Can change training time or episode length in `config.py`
