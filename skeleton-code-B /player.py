import copy
from queue import Queue
import numpy as np
import math
from collections import defaultdict
import sys
from referee.board import Board
import time
import random

_PLAYER_AXIS = {
    "red": 0, # Red aims to form path in r/0 axis
    "blue": 1 # Blue aims to form path in q/1 axis
}

_SWITCH_TURN = {
    "red": "blue",
    "blue": "red"
}

class Player:
    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        self.player = player
        self.state = Board(n)
        self.turn_num = 0
       
    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        self.turn_num = self.turn_num + 1

        # For first move, blue player swaps if red makes illegal move (center move) at the start
        # Otherwise, blue picks centre cell
        if self.player == "blue" and self.turn_num == 1:
            if self.illegal_start_move_by_red():
                return ("STEAL", )
            else:
                return ("PLACE", self.state.n // 2, self.state.n // 2)

        # For first move, red picks corner to avoid being stealed by blue
        if self.player == "red" and self.turn_num == 1:
            return ("PLACE", self.state.n-1, self.state.n-1)
        
        # Second turn onwards, run MCTS
        root = MonteCarloTreeSearchNode(root_color=self.player, curr_player = self.player, state=self.state)
        return root.best_action()
    
    def turn(self, player, action):
        """
        Called at the end of each player's turn to inform this player of 
        their chosen action. Update your internal representation of the 
        game state based on this. The parameter action is the chosen 
        action itself. 
        
        Note: At the end of your player's turn, the action parameter is
        the same as what your player returned from the action method
        above. However, the referee has validated it at this point.
        """
        _, r, q = action
        
        # Update state
        if _ == "PLACE":
            captures = self.state.place(player, (r, q))
            self.state.__setitem__((r, q), player)
            if captures:
                for capture in captures:
                    self.state.__setitem__(capture, None)
        elif _ == "STEAL":
            self.state.swap()

    def illegal_start_move_by_red(self):
        """
        Check whether red selects centre hex in its first turn
        """
        current_state = self.state
        count_red = 0
        for i in range(1, current_state.n-1):
            count_red = count_red + list(current_state._data[i]).count(1)
        
        if count_red == 1:
             # Red is placed in centre for first move
            return True
        return False

class MonteCarloTreeSearchNode:
    """
    Code adapted from https://ai-boson.github.io/mcts/
    """

    def __init__(self, root_color, curr_player, state, parent=None, parent_action=None):
        self.root_color = root_color
        self.curr_player = curr_player
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.number_of_visits = 0
        self.results = defaultdict(int)
        self.results[1] = 0
        self.results[-1] = 0
        self.available_actions = self.actions()

    def actions(self):
        """
        Return all possible actions for the state
        """
        available_actions = []
        current_state_data = self.state._data
        for r in range(len(current_state_data)):
            for q in range (len(current_state_data)):
                if current_state_data[(r, q)] == 0:
                    available_actions.append(("PLACE", r, q))
        random.shuffle(available_actions)
        return available_actions

    def best_action(self):
        """
        Return node with best move 
        """
        start = time.time()
        while time.time() - start < 1.92:
            node = self.tree_policy()
            result = node.simulation()
            node.back_propagate(result)
        return self.best_child().parent_action
        
    def tree_policy(self):
        """
        Return node to run simulation
        """
        current_node = self
        while not current_node.is_terminal(current_node.parent_action):
            if not len(current_node.available_actions) == 0:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def expand(self):
        """
        Generate a new child of selected node
        """
        action = self.available_actions.pop()
        state = self.play(action)
        child = MonteCarloTreeSearchNode(root_color=self.root_color, curr_player = self.curr_player, state=state, parent=self, parent_action=action)
        self.children.append(child)
        return child

    def best_child(self, C_params=math.sqrt(2)): 
        """
        Rank children based on UCB1 formula and return best child
        """
        weights = []
        for child in self.children:
            if child.N() == 0:
                weights.append(sys.maxsize)
                continue
            weights.append(child.U() / child.N() + C_params * math.sqrt(math.log(self.N()) / child.N()))
                  
        return self.children[np.argmax(weights)]

    def N(self):
        """
        Return number of playouts through node
        """
        return self.number_of_visits   
    
    def U(self):
        """
        Return total utility of all playouts that go through node (number of wins)
        """
        return self.results[1]

    def play(self, action):
        """
        Return a state that results from taking action
        """
        _, r, q = action
       
        # Update state
        resulting_state = copy.deepcopy(self.state)
        player = self.curr_player
        
        captures = resulting_state.place(player, (r, q))
        resulting_state.__setitem__((r, q), player)
        if captures:
            for capture in captures:
                resulting_state.__setitem__(capture, None)

        return resulting_state

    def simulation(self):
        """
        Perform simulation for a generated child
        """
        curr_node = self
        
        action = None

        while not curr_node.is_terminal(action):
            
            # Get all possible actions from the current state
            possible_actions = curr_node.actions()
            
            # Play the best action 
            action = curr_node.simulation_policy(possible_actions)
            state = curr_node.play(action)
            curr_node = MonteCarloTreeSearchNode(root_color=self.root_color, curr_player= _SWITCH_TURN[curr_node.curr_player], state=state, parent=curr_node, parent_action=action)

        return curr_node.game_result(curr_node.parent_action)


    def simulation_policy(self, possible_actions):
        """
        Choose best action out of possible actions available in simulation

        FOR NOW, we do randomly
        """
        return possible_actions[np.random.randint(len(possible_actions))]

    def back_propagate(self, result):
        """
        Update all nodes up to root
        """
        self.number_of_visits += 1
        self.results[result] += 1
        if self.parent:
            self.parent.back_propagate(result)

    def is_terminal(self, action):
        """
        Check whether a state is terminal state
        """
        return self.game_result(action) != None
          
        
    def game_result(self, action):
        """
        Function is adapted from referee
        Returns 1 for win, -1 for loss and 0 for tie, otherwise None 
        """
        curr_state = self.state
        player = self.root_color

        if not action:
            return None

        # Get the player who played last move
        last_player = _SWITCH_TURN[self.curr_player]

        # Check if path is formed on axis depending on player
        _, r, q = action
        reachable = curr_state.connected_coords((r,q))
        axis_vals = [coord[_PLAYER_AXIS[last_player]] for coord in reachable]
        if min(axis_vals) == 0 and max(axis_vals) == curr_state.n - 1:
            if player == last_player:
                return 1
            else:
                return -1

        return None 
