import logging
import os
import sys
from collections import deque
import pickle
from pickle import Pickler, Unpickler
import random
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS_bpp import MCTS

import wandb
import time

log = logging.getLogger(__name__)


class CoachBPP():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, items_list, total_area, gen, args, saved_rewards_list=[]):
        self.game = game
        self.args = args
        # self.items_list = items_list # the items to be packed (w, h, a, b) from the generator
        # self.items_list = [[1, 10, 0, 0], [4, 5, 0, 0], [4, 5, 0, 0], [4, 3, 0, 0], [1, 3, 0, 0], [3, 1, 0, 0], [2, 1, 0, 0], [5, 3, 0, 0], [3, 3, 0, 0], [2, 3, 0, 0]]
        self.items_list = [[4, 2, 0, 0], [3, 3, 0, 0], [2, 5, 0, 0], [1, 8, 0, 0], [1, 8, 0, 0], [1, 8, 0, 0], [3, 1, 0, 0], [4, 7, 0, 0], [3, 5, 0, 0], [1, 8, 0, 0]]
        self.items_total_area = total_area # given for simplicity; to be changed later TODO
        # Q: in BPP, we could generate different item sets. Do we treat each item set as a different problem? or do we change item sets during 
        # training?
        self.rewards_list = saved_rewards_list.copy()
        self.ep_score = 0
        self.mcts = MCTS(self.game, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        self.gen = gen
        random.seed(args.seed)
        self.seeds = random.sample(range(0, 10000), 500)

    def executeEpisode(self, greedy=False):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        board = self.game.getInitBoard()
        items_list_board = self.game.getInitItems(self.items_list)
        episodeStep = 0

        placement_info = {'placement': [],
                    'score': 0,}

        t0 = time.time()
        # estimate height
        bin_items_state = self.game.getBinItem(board, items_list_board)
        height_list = []
        size_list = []
        for i in range(len(self.items_list)):
            size_list.append(self.items_list[i][0]*self.items_list[i][1])
            height_list.append(self.items_list[i][1])
        h_e =  int(max(np.ceil(self.items_total_area / self.game.bin_width), max(height_list)))

        alpha_heuristics = []
        for i in range(h_e):
            total_ = 0
            for j in range(len(self.items_list)):
                if self.items_list[j][1] >= i+1:
                    total_ += self.items_list[j][0]
            alpha_heuristics.append(total_/len(self.items_list))

        Is = []
        for i in range(len(self.items_list)):
            tmp = [sum(items_list_board[i][t]) for t in range(h_e)]
            Is.append(tmp)

        while True:
            episodeStep += 1
            bin_items_state = self.game.getBinItem(board, items_list_board)

            remaining_cap = [(15 - sum(board[t]))/15 for t in range(h_e)]

            wdps = []
            for t in range(len(items_list_board)):
                if sum(sum(items_list_board[t])) == 0:
                    wdps.append(-1e5)
                    continue
                else:
                    wdp = sum([x*y*z for x,y,z in zip(alpha_heuristics,Is[t], remaining_cap)])
                    wdps.append(wdp)
            
            cur_item = np.argmax(wdps)

            # find placement
            valid_actions = self.game.getValidMoveForItem(bin_items_state, cur_item)
            as_ = []
            for ii in valid_actions:
                board_, _ = self.game.getNextState(board, ii, items_list_board)
                as_.append(self.game.get_minimal_bin_height(board_))
            action_chosen = valid_actions[np.argmin(as_)]

            item = items_list_board[cur_item]
            w = sum(item[0,:])
            h = sum(item[:,0])

            _, placement_ = int(action_chosen/self.game.bin_width), int(action_chosen%(self.game.bin_width))
            placement_info['placement'].append([placement_, w, h]) # y, x, w, h

            board, items_list_board = self.game.getNextState(board, action_chosen, items_list_board)
            next_bin_items_state = self.game.getBinItem(board, items_list_board)
            
            r, score = self.game.getGameEnded(next_bin_items_state, self.items_total_area, self.rewards_list, self.args.alpha)

            if r != 0:
                t_elapsed = time.time() - t0
                self.ep_score = score
                placement_info['score'] = score
                return placement_info, t_elapsed

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        eval_results = []
        t_total = 0
        for i in range(1, self.args.numIters + 1):
            log.info(f'Starting Game #{i} ...')
            # generate a new game
            np.random.seed(i-1)
            self.gen.bin_height = np.random.randint(self.args.binH_min, self.args.binH+1)
            self.items_total_area = self.gen.bin_height * self.gen.bin_width
            
            generator_seed = 0

            # seeds = [96890, 63470] # - two seeds used for self-play vs MCTS results visualization 
            for _ in tqdm(range(self.args.numEps), desc="Running MCTS for current game"):
                # generator_seed = np.random.randint(int(1e5))
                items_list = self.gen.items_generator(generator_seed)
                self.items_list = np.copy(items_list)
                generator_seed += 1
                # items_list = self.gen.items_generator(seeds[t])
                # generator_seed = np.random.randint(int(1e5))
                # items_list = self.gen.items_generator(generator_seed)
                # self.items_list = np.copy(items_list)
                # define self.items_list
                placement_info, t_elapsed = self.executeEpisode()
                eval_results.append(placement_info)
                self.rewards_list.append(self.ep_score)
                wandb.log({"all scores": self.ep_score})
                t_total += t_elapsed

            # print('reward buffer for ranked reward: ', self.rewards_list)                
            wandb.log({"final score for each game": self.ep_score}, step=i)
            # percentage_optim = sum([item == 1.0 for item in ep_scores]) / len(ep_scores)
            # wandb.log({"optimality percentage": percentage_optim,
            #             "min reward": np.min(ep_scores),
            #             "max reward": np.max(ep_scores)}, step=i)
            # print('-----------mean reward of Iter ', i, 'is-----------', np.mean(ep_scores))
            
            # save self.rewards_list in each iter
            # self.save_rewards_list()
        with open('eval_results_hvraa.pkl', 'wb') as f:
            pickle.dump(eval_results, f)

        print('average elapsed time', t_total/self.args.numIters)
        
    def save_rewards_list(self):
        file_n = 'rewards_list_' + str(self.args.numItems) + '_items.pkl'
        filepath = os.path.join(self.args.checkpoint, file_n)
        with open(filepath, 'wb') as f:
            pickle.dump(self.rewards_list, f)
    
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def arena_playing(self, pmcts, nmcts, seeds_iter):
        # if mean(scores)_n > mean(scores)_p, return 1, accept new model
        p_scores = []
        n_scores = []
        # seeds for arena playing
        random.seed()
        arena_seeds = random.sample(seeds_iter, self.args.arenaCompare)

        for t in tqdm(range(self.args.arenaCompare), desc="Arena playing"):
            # generate game
            # np.random.seed()
            # generator_seed = np.random.randint(int(1e5))
            generator_seed = arena_seeds[t]        
            items_list = self.gen.items_generator(generator_seed)
            items_list_p = np.copy(items_list)
            items_list_n = np.copy(items_list)
            
            # pmcts
            board = self.game.getInitBoard()
            items_list_board = self.game.getInitItems(items_list_p)
            bin_items_state = self.game.getBinItem(board, items_list_board)

            game_ended = 0

            while game_ended == 0:
                # choose action greedily
                pi = pmcts.getActionProb(bin_items_state, self.items_total_area, self.rewards_list, greedy_a=0)
                action = np.random.choice(len(pi), p=pi)

                board, items_list_board = self.game.getNextState(board, action, items_list_board)
                next_bin_items_state = self.game.getBinItem(board, items_list_board)
                bin_items_state = next_bin_items_state
                game_ended, score = self.game.getGameEnded(bin_items_state, self.items_total_area, self.rewards_list, self.args.alpha)
            p_scores.append(score)

            #nmcts
            board = self.game.getInitBoard()
            items_list_board = self.game.getInitItems(items_list_n)
            bin_items_state = self.game.getBinItem(board, items_list_board)

            game_ended = 0
            while game_ended == 0:
                # choose action greedily
                pi = nmcts.getActionProb(bin_items_state, self.items_total_area, self.rewards_list, greedy_a=0)
                action = np.random.choice(len(pi), p=pi)

                board, items_list_board = self.game.getNextState(board, action, items_list_board)
                next_bin_items_state = self.game.getBinItem(board, items_list_board)
                bin_items_state = next_bin_items_state
                game_ended, score = self.game.getGameEnded(bin_items_state, self.items_total_area, self.rewards_list, self.args.alpha)
            n_scores.append(score)

        # percentage_optim_n = sum([item == 1.0 for item in n_scores]) / len(n_scores)
        # percentage_optim_p = sum([item == 1.0 for item in p_scores]) / len(p_scores)

        if np.mean(n_scores) >= np.mean(p_scores):
            return 1
        else:
            return 0
