﻿import logging
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

    def __init__(self, game, nnet, items_list, total_area, gen, args, saved_rewards_list=[]):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(self.game, self.args)  # the competitor network
        self.items_list = items_list # the items to be packed (w, h, a, b) from the generator
        self.items_total_area = total_area # given for simplicity; to be changed later TODO
        # Q: in BPP, we could generate different item sets. Do we treat each item set as a different problem? or do we change item sets during 
        # training?
        self.rewards_list = saved_rewards_list.copy()
        # # 0918: initialize self.rewards_list
        # for i in range(25):
        #     self.rewards_list.append(1.0)
        self.ep_score = 0
        self.mcts = MCTS(self.game, self.nnet, self.args)
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
        trainExamples = []
        board = self.game.getInitBoard()
        items_list_board = self.game.getInitItems(self.items_list)
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            bin_items_state = self.game.getBinItem(board, items_list_board)
            if greedy:
                pi = self.mcts.getActionProb(bin_items_state, self.items_total_area, self.rewards_list, greedy_a=0)
            else:
                pi = self.mcts.getActionProb(bin_items_state, self.items_total_area, self.rewards_list)
            
            trainExamples.append([bin_items_state, pi, None])            
            # sym = self.game.getSymmetries(board, pi)
            # for b, p in sym:
            #     state_sym = self.game.getBinItem(b, items_list_board)
            #     trainExamples.append([state_sym, p, None])
            
            np.random.seed()
            action = np.random.choice(len(pi), p=pi)
            board, items_list_board = self.game.getNextState(board, action, items_list_board)
            next_bin_items_state = self.game.getBinItem(board, items_list_board)
            
            r, score = self.game.getGameEnded(next_bin_items_state, self.items_total_area, self.rewards_list, self.args.alpha)

            if r != 0:
                # self.rewards_list.append(score)
                # # if score  = [], does len(self.rewards_list) change?
                # if len(self.rewards_list) > self.args.numScoresForRank:
                #     self.rewards_list.pop(0)
                self.ep_score = score
                return [(x[0], x[1], r) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            ep_scores = []
            # store seeds in this iter
            seeds_iter = []
            # re-init a generator: for different height
            np.random.seed(i-1)
            self.gen.bin_height = np.random.randint(self.args.binH_min, self.args.binH+1)
            self.items_total_area = self.gen.bin_height * self.gen.bin_width

            generator_seed = 0
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    # 1. re-generate items: different game
                    # np.random.seed()
                    # generator_seed = np.random.randint(int(1e5))
                    items_list = self.gen.items_generator(generator_seed)
                    generator_seed += 1
                    seeds_iter.append(generator_seed)
                    self.items_list = np.copy(items_list)

                    iterationTrainExamples += self.executeEpisode(i>self.args.iterStepThreshold)
                    ep_scores.append(self.ep_score)

                # print('reward buffer for ranked reward: ', self.rewards_list)                
                wandb.log({"iter mean reward": np.mean(ep_scores)}, step=i)
                percentage_optim = sum([item == 1.0 for item in ep_scores]) / len(ep_scores)
                wandb.log({"optimality percentage": percentage_optim,
                            "min reward": np.min(ep_scores),
                            "max reward": np.max(ep_scores)}, step=i)
                # print('-----------mean reward of Iter ', i, 'is-----------', np.mean(ep_scores))
                
                # save the iteration examples to the history 
                # this is the examples used for training
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # if i % 100 == 0:  
            #     self.saveTrainExamples(i - 1)
            self.saveTrainExamples(i-1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            # nmcts = MCTS(self.game, self.nnet, self.args)

            # if not self.skipFirstSelfPlay or i > 1:
            #     log.info('PITTING AGAINST PREVIOUS VERSION')
            #     n_win = self.arena_playing(pmcts, nmcts, seeds_iter)
            # else:
            #     # if trained on saved samples in the first iter, accept new model by default
            #     n_win = 1

            # log.info('WIN: %d' % (n_win))
            # if n_win == 0:
            #     log.info('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     log.info('ACCEPTING NEW MODEL')
            #     # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            
            # save self.rewards_list in each iter
            self.save_rewards_list()
        
    def save_rewards_list(self):
        file_n = 'rewards_list_' + str(self.args.numItems) + '_items.pkl'
        filepath = os.path.join(self.args.checkpoint, file_n)
        with open(filepath, 'wb') as f:
            pickle.dump(self.rewards_list, f)
    
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + '.pth.tar'

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

    def execute_ep_eval(self, items_list):
        """
        This function executes one episode of bin packing.
        Returns:
        """
        board = self.game.getInitBoard()
        items_list_board = self.game.getInitItems(items_list)
        episodeStep = 0

        placement_info = {'placement': [],
                            'score': 0,}


        board = self.game.getInitBoard()
        items_list_board = self.game.getInitItems(items_list)
        self.curPlayer = 1
        episodeStep = 0

        t0 = time.time()
        while True:
            episodeStep += 1
            bin_items_state = self.game.getBinItem(board, items_list_board)
            pi = self.mcts.getActionProb(bin_items_state, self.items_total_area, self.rewards_list, greedy_a=0)           
            np.random.seed()
            action = np.random.choice(len(pi), p=pi)
            # save action --> placement
            cur_item, placement = int(action/self.game.bin_width), int(action%(self.game.bin_width))
            item = items_list_board[cur_item]
            assert sum(sum(item)) > 0 # must choose a valid item
            # item is valid
            w = sum(item[0,:])
            h = sum(item[:,0])

            placement_info['placement'].append([placement, w, h]) # y, x, w, h

            # execute action
            board, items_list_board = self.game.getNextState(board, action, items_list_board)
            next_bin_items_state = self.game.getBinItem(board, items_list_board)
            r, score = self.game.getGameEnded(next_bin_items_state, self.items_total_area, self.rewards_list, self.args.alpha)


            if r != 0:
                t_elapsed = time.time() - t0
                # self.rewards_list.append(score)
                # # if score  = [], does len(self.rewards_list) change?
                # if len(self.rewards_list) > self.args.numScoresForRank:
                #     self.rewards_list.pop(0)
                placement_info['score'] = score
                return placement_info, t_elapsed  
    
    def run_eps_save(self):
        """Run eps using loaded model, save results for visualization.
        Returns:
            None. Evaluation results (eval_results) is pickled.
        """
        log.info(f'Starting Evaluation')

        eval_results = []
        generator_seed = 0
        t_total = 0
        for i in tqdm(range(self.args.numEps), desc="Self Play"):
            self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
            # 1. re-generate items: different game
            np.random.seed(i)
            self.gen.bin_height = np.random.randint(self.args.binH_min, self.args.binH+1)
            self.items_total_area = self.gen.bin_height * self.gen.bin_width

            # generator_seed = np.random.randint(int(1e5))
            items_list = self.gen.items_generator(generator_seed)
            generator_seed += 1
            items_list_eval = np.copy(items_list)
            # # 2. same game (items shapes fixed)
            # items_list = self.gen.items_generator(self.args.seed)
            # self.items_list = np.copy(items_list)

            placement_info, t_elapsed = self.execute_ep_eval(items_list_eval)
            eval_results.append(placement_info)
            t_total += t_elapsed
            # # use the same func with training; only record scores
            # self.items_list = np.copy(items_list_eval)
            # placement_info = {'placement': [],
            #         'score': 0,}
            
            # self.executeEpisode()
            # placement_info['score'] = self.ep_score
            # eval_results.append(placement_info)


            with open('eval_results.pkl', 'wb') as f:
                pickle.dump(eval_results, f)

        print('average elapsed time', t_total/self.args.numEps)
