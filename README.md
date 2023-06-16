
<div id="top"></div>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ay-ka/Reinforcement-Learning">
    <img style="float:left" src="images/rl.jpg" alt="reinforcement learning" width="700" height="300">
  </a>
  <h6 align="center"; display: flex; justify-content: center>Implementation of MADDPG (Multi-Agent Deep Deterministic Policy Gradient) and QMIX (Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning) as well as combining these RL algorithms with EA algorithms (CMAES, CEM, NEAT) & applying to different Benchmark such as Robosuite (Robotic Manipulator Benchmark) </h6>
</div>

<br />
<br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Prerequisites-Installation">Prerequisites & Installation</a></li>
        <li><a href="#How-To-Run">How To Run</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project



:star:<b> Algorithms used in this project </b>

<ul>
  <li>
    <b><a href="https://arxiv.org/abs/1706.02275">MADDPG (Multi-Agent Deep Deterministic Policy Gradient)</a>:</b> MADDPG, or Multi-agent DDPG, extends                                                       DDPG into a multi-agent policy gradient algorithm where decentralized agents learn a centralized critic based on the observations and actions of all 

agents.
  </li
  <li>
     <b><a href="https://arxiv.org/abs/1803.11485">QMIX (Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning)</a>:</b> QMIX is a multi agent deep reinforcement learning methods based on Q-learning and value-defactorazations; novel value-based method that can train decentralised policies in a centralised end-to-end fashion
  </li>
  <li>
    <b><a href="https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf">NEAT (NeuroEvolution of Augmenting Topology)</a>:</b> NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm that creates artificial neural networks
  </li>
  <li>
    <b><a href="https://link.springer.com/article/10.1007/s10479-005-5724-z">CEM (Cross Enthropy Method)</a>: </b> The cross-entropy method is a versatile heuristic tool for solving difficult estimation and optimization problems, based on Kullback–Leibler (or cross-entropy) minimization.
  </li>
  <li>
    <b><a href="https://en.wikipedia.org/wiki/CMA-ES">CMA-ES (Covariance Matrix Adaption - Evolutionary Strategies): (PSO)</a>:</b> The  CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is an evolutionary algorithm for difficult non-linear non-convex black-box optimisation problems in continuous domain
  </li>
  <li>
    <b>Genetic:</b> A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation. 
  </li>
  <br/>
</ul>




:star:<b> Describe Implemented projects</b>

<ul>
  <li>
     <b>ERL_MADDPG:</b> This project is about Implementing <b>MADDPG</b> and combining it with <b>Genetic Algorithm</b> based on method proposed on <a        href="https://arxiv.org/abs/1805.07917">Evolution-Guided Policy Gradient In Reinforcement Learning</a>, apllied on different benchmarks
  </li>
  <li>
     <b>ROBOSUITE_MADDPG:</b> This project is about Implementing <b>MADDPG</b> and combining it with <b>Genetic Algorithm</b> based on method proposed on <a        href="https://arxiv.org/abs/1805.07917">Evolution-Guided Policy Gradient In Reinforcement Learning</a>, apllied on different benchmarks
  </li>
  <li>
    <b>NEAT_QMIX:</b> This project is about Implementing <b>QMIX</b> and combining with <b>NEAT</b> taken from <a                                                                href="https://neat-python.readthedocs.io">Repo</a> and evaluate on different benchmarks
  </li>

  <li>
    <b>CEM-MADDPG:</b> This project is about Implementing <b>MADDPG</b> and <b>CEM</b> Algorithms as well as combining these two algorithms; and evaluating on different benchmarks (this project need future improvements)
  </li>
  <li>
     <b>CMAES_MADDPG:</b> This project is about Implementing <b>MADDPG</b> and <b>CAM-ES</b> Algorithms as well as combining these two                                       algorithms and evaluating on different benchmarks (this project need future improvements)
  </li>
</ul>


<p align="right">(<a href="#top">back to top</a>)</p>




