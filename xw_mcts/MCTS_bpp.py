import logging
import math

import numpy as np
import random

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def get_best_action(self, canonicalBoard, totalArea, rewardsList, greedy_a=0):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            action: the best action picked my MCTS.
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, totalArea, rewardsList)

        # find best action
        s = self.game.stringRepresentation(canonicalBoard)
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        for a in range(self.game.getActionSize()):
            if valids[a]:
                u = self.Qsa[(s, a)]
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        if greedy_a == 0:
            # choose action greedily
            probs = [0] * self.game.getActionSize()
            probs[a] = 1
            return probs
        return probs

    def search(self, canonicalBoard, totalArea, rewardsList):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s], _ = self.game.getGameEnded(canonicalBoard, totalArea, rewardsList, self.args.alpha)
            
        if self.Es[s] != 0:
            # terminal node
            return self.Es[s]

        if s not in self.Ns:
            # leaf node, run Monte-Carlo rollout
            # return win or lose
            valids = self.game.getValidMoves(canonicalBoard)
            v = self.rollout_policy(canonicalBoard, totalArea, rewardsList)

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]

        # pick the action with the highest upper confidence bound
        a_to_choose = []
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if not (s, a) in self.Qsa: # this action has not been choosen
                    a_to_choose.append(a)
        if len(a_to_choose) > 0:
            # some actions are not explored yet, choose one of them
            a = random.choice(a_to_choose)
        else:
            # calculate UCB
            cur_best = -float('inf')
            best_act = -1
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    u = self.Qsa[(s, a)] + self.args.c_mcts * math.sqrt(math.log(self.Ns[s]) / self.Nsa[(s, a)])
                    if u > cur_best:
                        cur_best = u
                        best_act = a
            a = best_act

        board, items_list_board = self.game.getNextState(canonicalBoard[0], a, canonicalBoard[1:])
        next_bin_items_state = self.game.getBinItem(board, items_list_board)

        v = self.search(next_bin_items_state, totalArea, rewardsList)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

    def rollout_policy(self, canonicalBoard, totalArea, rewardsList):
        game_end = 0
        all_actions = list(range(self.game.getActionSize()))
        while game_end == 0:            
            valids = self.game.getValidMoves(canonicalBoard)
            valids = valids / sum(valids) # normalize
            a = np.random.choice(all_actions, p=valids)
            board, items_list_board = self.game.getNextState(canonicalBoard[0], a, canonicalBoard[1:])
            canonicalBoard = self.game.getBinItem(board, items_list_board)
            game_end, _ = self.game.getGameEnded(canonicalBoard, totalArea, rewardsList, self.args.alpha)
        return game_end    