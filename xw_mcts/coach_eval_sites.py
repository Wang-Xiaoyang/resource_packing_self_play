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

    def __init__(self, game, nnet, items_list, total_area, gen, args, cpu_list, saved_rewards_list=[]):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(self.game, self.args)  # the competitor network
        self.items_list = items_list # the items to be packed (w, h, a, b) from the generator
        self.items_total_area = total_area # given for simplicity; to be changed later TODO
        # Q: in BPP, we could generate different item sets. Do we treat each item set as a different problem? or do we change item sets during 
        # training?
        self.rewards_list = saved_rewards_list.copy()
        self.ep_score = 0
        self.mcts = MCTS(self.game, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        self.gen = gen
        # random.seed(args.seed)
        self.seeds = random.sample(range(0, 10000), 500)
        self.cpu_list = cpu_list
        
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

    def execute_ep_eval(self, greedy=False):
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
        self.curPlayer = 1
        episodeStep = 0

        placement_info = {'placement': [],
                    'score': 0,}

        while True:
            episodeStep += 1
            bin_items_state = self.game.getBinItem(board, items_list_board)
            pi = self.mcts.get_best_action(bin_items_state, self.items_total_area, self.rewards_list)
            action = np.random.choice(len(pi), p=pi)
            # save action --> placement
            cur_item, placement = int(action/self.game.bin_width), int(action%(self.game.bin_width))
            item = items_list_board[cur_item]
            assert sum(sum(item)) > 0 # must choose a valid item
            # item is valid
            w = sum(item[0,:])
            h = sum(item[:,0])

            placement_info['placement'].append([placement, w, h]) # y, x, w, h

            board, items_list_board = self.game.getNextState(board, action, items_list_board)
            next_bin_items_state = self.game.getBinItem(board, items_list_board)
            
            r, score = self.game.getGameEnded(next_bin_items_state, self.items_total_area, self.rewards_list, self.args.alpha)

            if r != 0:  
                self.ep_score = score
                placement_info['score'] = score
                return placement_info 
    
    def run_eps_save(self):
        """Run eps using loaded model, save results for visualization.

        Returns:
            None. Evaluation results (eval_results) is pickled.
        """
        log.info(f'Starting Evaluation')

        eval_results = []

        generator_seed = 100
        for _ in tqdm(range(self.args.numEps), desc="Self Play"):
            self.mcts = MCTS(self.game, self.args)  # reset search tree
            # 1. re-generate items: different game
            # generator_seed = np.random.choice(self.seeds)
            # np.random.seed()
            # generator_seed = np.random.randint(int(1e5))
            max_w = 6 # 6 for du2; 13 for du1
            items_list = self.gen.items_generator_set_one_dim(generator_seed, max_w, self.cpu_list, mode='random')
            generator_seed += 1
            # items_list = self.gen.items_generator(generator_seed)
            self.items_list = np.copy(items_list)
            self.items_total_area = 0
            for i in range(len(items_list)):
                self.items_total_area += items_list[i][1] * items_list[i][0]

            # self.rewards_list = [self.items_total_area/((np.ceil(np.sqrt(self.items_total_area))**2) + (np.floor(np.sqrt(self.items_total_area))**2))/2 for i in range(100)]
            # self.rewards_list = [1 for i in range(100)]

            # # 2. same game (items shapes fixed)
            # items_list = self.gen.items_generator(self.args.seed)
            # self.items_list = np.copy(items_list)

            eval_results.append(self.execute_ep_eval())
            # # use the same func with training; only record scores
            # self.items_list = np.copy(items_list_eval)
            # placement_info = {'placement': [],
            #         'score': 0,}
            
            # self.executeEpisode()
            # placement_info['score'] = self.ep_score
            # eval_results.append(placement_info)


            with open('eval_results_real_sites_du2_mcts.pkl', 'wb') as f:
                pickle.dump(eval_results, f)