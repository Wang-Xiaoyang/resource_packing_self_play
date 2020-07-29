from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .BinPackingLogic import Bin
import numpy as np

class BinPackingGame(Game):
    # square_content = {
    #     -1: "X",
    #     +0: "-",
    #     +1: "O"
    # }

    def __init__(self, bin_width, bin_height, num_items, n):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.num_items = num_items
        self.n = n # number of bins (consider later - xw)
        self.cur_item = 0 # counter(idx) for item(s) being considered

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Bin(self.bin_width, self.bin_height)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.bin_height, self.bin_width)

    def getActionSize(self):
        # return number of actions
        # total number of actions in this board
        return (self.bin_height*self.bin_width)*self.num_items + self.num_items

    def getInitItems(self, items_list):
        # items_list from item generator
        items_list_board = []
        for i in range(self.num_items):
            w, h, _, _ = items_list[i]
            item_board = self.getInitBoard()
            item_board[0:h, 0:w] = 1
            items_list_board += [item_board]
        return items_list_board
    
    def getItemsUpdated(self, items_list_board, cur_item):
        # update items without touching the board (bin).
        items_list_board[cur_item] -= items_list_board[cur_item] # set to 0     
        return items_list_board

    def getNextState(self, board, action, items_list_board):
        # get next board, to see if game ended - xw
        # also the next state to keep game going!

        # if player takes action on board, return next (board,player)
        # action must be a valid move
        items_list_board = np.copy(items_list_board)
        b = Bin(self.bin_width, self.bin_height)
        b.pieces = np.copy(board)
        cur_item, placement = int(action/(self.bin_height*self.bin_width)), action%(self.bin_height*self.bin_width)
        if cur_item == self.num_items: # pass item or do nothing
            if placement == self.num_items:
                return b.pieces, items_list_board
            else:
                items_list_board = self.getItemsUpdated(items_list_board, placement)
                return b.pieces, items_list_board
        item = items_list_board[cur_item] # board format
        assert sum(sum(item)) > 0 # must choose a valid item
        # item is valid
        w = sum(item[0,:])
        h = sum(item[:,0])
        move = (int(placement/self.bin_width), placement%self.bin_width)
        b.execute_move(move, w, h)
        items_list_board = self.getItemsUpdated(items_list_board, cur_item)
        return (b.pieces, items_list_board)

    def getValidMoves(self, board):
        # return a fixed size binary vector
        # size is the same with getActionSize; the value is 1 for valid moves in the 'board'
        valids = [0]*self.getActionSize()
        b = Bin(self.bin_width, self.bin_height)
        b.pieces = np.copy(board[0])
        pass_valids = [0] * self.num_items
        legal_moves = []
        for item in range(self.num_items):
            if sum(sum(board[item+1])) == 0:
                continue
            legal_moves += b.get_moves_for_square(board[1:], item)
            pass_valids[item] = 1
        if len(legal_moves)==0:
            valids[-self.num_items:] = pass_valids
            return np.array(valids)
        for item, x, y in legal_moves:
            valids[(item*(self.bin_height*self.bin_width)+x*self.bin_width+y)] = 1
            valids[-self.num_items:] = pass_valids
        return np.array(valids)

    def getGameEnded(self, total_board, items_total_area, rewards_list, alpha):
        # return 0 if not ended, 1 if win (higher than 0.75 reward), -1 if lost
        assert(len(total_board) == self.num_items+self.n)
        if sum(sum(sum(total_board[1:]))) != 0:
            return 0, []
        else:
            return self.getRankedReward(total_board, items_total_area, rewards_list, alpha)

    def getBinItem(self, board, items_list_board):
        # get the state: bin representation + items representation
        return np.array([board] + list(items_list_board))

    def getSymmetries(self, board, pi):
        # rotate 180 degree; flip in two ways
        # note: add one layer in the state indicating the current item
        # to keep the pi simple

        # mirror, rotational
        assert(len(pi) == self.getActionSize())  # 1 for pass
        size_b = self.bin_width * self.bin_height
        pi_board = []
        for item in range(self.num_items):
            pi_board_ = np.reshape(pi[item*size_b:(item+1)*size_b], (self.bin_height, self.bin_width))
            pi_board.append(pi_board_)
        l = []
        for i in [2, 4]:
            newB = np.rot90(board, i)
            newPi = []
            pi_board_c = pi_board.copy()
            for item in range(self.num_items):
                pi_board_ = pi_board_c[item]
                newPi_ = np.rot90(pi_board_, i)
                newPi += list(newPi_.ravel())
            l += [(newB, list(np.copy(newPi)) +  pi[-self.num_items:])]
        for i in [2, 4]:
            newB = np.rot90(board, i)
            newB = np.fliplr(newB)
            newPi = []
            pi_board_c = pi_board.copy()
            for item in range(self.num_items):
                pi_board_ = pi_board_c[item]
                newPi_ = np.rot90(pi_board_, i)
                newPi_ = np.fliplr(newPi_)
                newPi += list(newPi_.ravel())
            l += [(newB, list(np.copy(newPi)) +  pi[-self.num_items:])]
        # duplicated!
        # for i in [2, 4]:
        #     newB = np.rot90(board, i)
        #     newB = np.flipud(newB)
        #     newPi = []
        #     pi_board_c = pi_board.copy()
        #     for item in range(self.num_items):
        #         pi_board_ = pi_board_c[item]
        #         newPi_ = np.rot90(pi_board_, i)
        #         newPi_ = np.flipud(newPi_)
        #         newPi += list(newPi_.ravel())
        #     l += [(newB, list(np.copy(newPi)) +  pi[-self.num_items:])]
        return l

    def get_minimal_bin(self, board):
        for i in reversed(range(self.bin_height)):
            if sum(board[i,:]) > 0:
                break
        h = i + 1
        for j in reversed(range(self.bin_width)):
            if sum(board[:,j]) > 0:
                break
        w = j + 1
        a = max([h, w])
        return a
    
    def getRankedReward(self, total_board, items_total_area, rewards_list, alpha):
        # alpha: the ranked reward parameter
        rewards_list = rewards_list.copy()

        if sum(sum(total_board[0,:])) != items_total_area:
            # some items are discarded instead of being placed in the bin
            r = 0
        else:
            a = self.get_minimal_bin(total_board[0,:])
            r = items_total_area / (a*a)
        # r = sum(sum(total_board[0,:])) / items_total_area

        if len(rewards_list) == 0:
            return 1, r
        sorted_reward = np.sort(rewards_list)
        bl = sorted_reward[int(np.floor(len(sorted_reward) * alpha))-1]
        if r > bl or r == 1:
            return 1, r
        elif r < bl:
            return -1, r
        else:
            return np.random.choice([1, -1], p=[0.5, 0.5]), r

    def stringRepresentation(self, board):
        board_ = np.array([]).tostring()
        for i in board:
            board_ += i.tostring()
        return board_

    def stringRepresentationReadable(self, board):
        # not used
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        #? needed?
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        # later - xw
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")


class ItemsGenerator():

    def __init__(self, bin_width, bin_height, items):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.n = items

    def items_generator(self, seed):
        np.random.seed(seed)
        item_list = [[self.bin_width, self.bin_height, 0, 0]] # initial item equals to the bin

        while len(item_list) < self.n:
            axis = np.random.randint(2) # 0 for x , 1 for y axis
            idx_item = np.random.randint(len(item_list)) # choose an item to split
            [w, h, a, b] = item_list[idx_item]
            if axis == 0:
                if w == 1:
                    continue
                x_split = np.random.randint(a+1, a+w)
                new_w = x_split - a
                item_s1 = [new_w, h, a, b]
                item_list.append(item_s1)
                item_s2 = [w-new_w, h, x_split, b]
                item_list.append(item_s2)
                item_list.pop(idx_item)
            elif axis == 1:
                if h == 1:
                    continue
                y_split = np.random.randint(b+1, b+h)
                new_h = y_split - b
                item_s1 = [w, new_h, a, b]
                item_list.append(item_s1)
                item_s2 = [w, h-new_h, a, y_split]
                item_list.append(item_s2)
                item_list.pop(idx_item)
        return item_list