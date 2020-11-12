import logging

import coloredlogs

from CoachBPP import CoachBPP as Coach
from binpacking.BinPackingGame import BinPackingGame as Game
from binpacking.BinPackingGame import ItemsGenerator as Generator

# from binpacking.pytorch.NNet import NNetWrapper as nn
from utils import *

import torch
import wandb
import pickle

wandb.init(entity="xiaoyang",
            project="resource-packing-self-play")
# wandb config parameters
wandb.config.binW, wandb.config.binH = 15, 10
wandb.config.binH_min = 7
wandb.config.virtual_bin_w, wandb.config.virtual_bin_h = 15, 15
wandb.config.numItems, wandb.config.numBins = 10, 1
wandb.config.numIters = 100 #total games
wandb.config.numEps = 2 # repeat #times for each game
wandb.config.iterStepThreshold = 50  # choose actions greedily after # iters in training; exploration vs exploitation
wandb.config.updateThreshold = 0.6
wandb.config.maxlenOfQueue = 200000
wandb.config.numMCTSSims = 200 #100
wandb.config.arenaCompare = 20 #20; for each agent
wandb.config.c_mcts = 2 # UCT parameter in MCTS
wandb.config.alpha = 0.75
wandb.config.seed = 100
wandb.config.numItersForTrainExamplesHistory = 200 # keep training samples for # iters
wandb.config.numScoresForRank = 100 # total number of saved reward (for ranked reward)
wandb.config.lr = 0.001
wandb.config.dropout = 0.1
wandb.config.epochs = 10
wandb.config.batch_size = 64
wandb.config.cuda = torch.cuda.is_available()
wandb.config.num_channels = 256 # 512
wandb.config.nnet_type = 'ResNet'
wandb.config.load_model = False
wandb.config.load_folder_file = ('/xw_mcts_wild_multi_game_10/temp','best.pth.tar')
wandb.config.load_rewards_list = False
wandb.config.load_rewards_list_file = '/home/xiaoyang/Documents/resource_packing_self_play/complete_runs/run-20201108_220640-m3c7x7l3/temp/rewards_list_10_items.pkl'
config = wandb.config

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# Training args
args = dotdict({
    'numIters': config.numIters,
    'numEps': config.numEps,              # Number of complete self-play games to simulate during a new iteration.
#    'epStepThreshold': config.epStepThreshold,        #
    'updateThreshold': config.updateThreshold,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': config.maxlenOfQueue,    # Number of game examples to train the neural networks.
    'numMCTSSims': config.numMCTSSims,          # Number of games moves for MCTS to simulate.
    'arenaCompare': config.arenaCompare,         # Number of games to play during arena play to determine if new net will be accepted.
    'c_mcts': config.c_mcts,
    'alpha': config.alpha,
    'seed': config.seed,
    'numScoresForRank': config.numScoresForRank,
    'numItems': config.numItems,
    'numBins': config.numBins,
    'iterStepThreshold': config.iterStepThreshold,
    'binH_min': config.binH_min,
    'binH': config.binH,

    'lr': config.lr,
    'dropout': config.dropout,
    'epochs': config.epochs,
    'batch_size': config.batch_size,
    'cuda': config.cuda,
    'num_channels': config.num_channels,
    'num_items': config.numItems,
    'num_bins': config.numBins,

    'checkpoint': './temp/',
    'load_model': config.load_model,
    'load_folder_file': config.load_folder_file,
    'load_rewards_list': config.load_rewards_list,
    'load_rewards_list_file': config.load_rewards_list_file,
    'numItersForTrainExamplesHistory': config.numItersForTrainExamplesHistory,
})


def main():
    log.info('Loading %s...', Game.__name__)
    # g = Game(config.binW, config.binH, config.numItems, config.numBins)
    g = Game(config.virtual_bin_w, config.virtual_bin_h, config.numItems, config.numBins)
    gen = Generator(config.binW, config.binH, config.numItems)
    items_list = gen.items_generator(args.seed)

    # log.info('Loading %s...', nn.__name__)
    # nnet = nn(g, args)

    # if args.load_model:
    #     log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    # else:
    #     log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    if args.load_rewards_list:
        log.info('Loading reward list file "%s"...', args.load_rewards_list_file)
        with open(args.load_rewards_list_file, 'rb') as f:
            saved_rewards_list = pickle.load(f)
        c = Coach(g, items_list, (config.binW*config.binH), gen, args, saved_rewards_list=saved_rewards_list)
    else:
        log.info('Not laoding reward list file')
        c = Coach(g, items_list, (config.binW*config.binH), gen, args)

    # c = Coach(g, items_list, (config.binW*config.binH), gen, args)   

    # if args.load_model:
    #     log.info("Loading 'trainExamples' from file...")
    #     c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

    # # save trained model:
    # wandb.save(args.checkpoint + 'temp.pth.tar')
    # wandb.save(args.checkpoint + 'best.pth.tar')

if __name__ == "__main__":
    main()
