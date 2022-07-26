# Artificial Intelligence (COMP30024) 2022 Semester 1 Assignment 2

# Project:
Design and implement a program to play the game of Cachex. That is, given information about the evolving state of the game, the program will decide on an action
to take on each of its turns. 

# Cachex Game Short Intro
Given a Hexagonal grid, a player wins the game if he/she forms a line from one end to the other. 

# Implementation:
- Used Monte Carlo Tree Search for selecting the most optimal action
    - Selection: From the root node, we chose a move that has the most weight which is calculated by UCB1, repeat the process down to the leaf.
    - Expansion: Generate all possible children of a selected node
    - Simulation: Perform playout from generated child node, moves by each player will be chosen randomly given the available actions that player can make
    - Backpropagation: Use the utility which is calculated at the end of the simulation to update the tree nodes up to the root. The utility is the number of wins.
    
(Due to time constraints, the program was not able to fully utilise the monte carlo search and did not perform well in the evaluation stage)



