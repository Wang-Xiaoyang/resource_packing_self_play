'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

"""
Inherited from OthelloLogic.py, for bin configuration in bin packing problem.
"""
import numpy as np

class Bin():

    # # list of all 8 directions on the board, as (x,y) offsets
    # __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, bin_width, bin_height):
        "Set up initial bin configuration."
        self.bin_width = bin_width
        self.bin_height = bin_height
        # Create the empty bin array, height * width
        self.pieces = [None]*self.bin_height
        for i in range(self.bin_height):
            self.pieces[i] = [0]*self.bin_width

    # add [][] indexer syntax to the Bin
    def __getitem__(self, index): 
        return self.pieces[index]

    def count_pieces(self, color):
        """Counts the # pieces of the given status
        (1 for occupied, 0 for empty spaces)"""
        count = 0
        for y in range(self.bin_width):
            for x in range(self.bin_height):
                if self[x][y]==color:
                    count += 1
        return count
    
    def get_legal_moves(self):
        #?
        # if an item is already places, this move is not legal anymore 
        """Returns all the legal moves for the current bin and items state.

        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        # check if all items placed?
        # check if game ends; all places items are marked as 0 (in state representation)

        # if there's no legal moves left, game ended
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    # def get_moves_for_square(self, item_idx):
    #     # get moves for each available item -xw
    #     # then get all valid moves in get_legal_moves
    #     item = items_all[item_idx] # board format
    #     w, h, _, _ = items_info[item_idx]
    #     moves = []
    #     if sum(item) != 0:
    #         # current item to be placed
    #         for i in range(self.bin_width-w+1):
    #             for j in range(self.bin_height-h+1):
    #                 if sum(self[j:j+h][i:i+w]) == 0:
    #                     moves.append((item_idx, i, j))
    #     return moves

    def get_adjacency(self, i, j, h, w):
        adjacent_direction = 0
        # get adjacency information for current item (h, w) if placed in (i, j)
        # up
        if i == 0:
            adjacent_direction += 1
        else:
            if sum(self[i-1, j:j+w]) != 0:
                adjacent_direction += 1
        # down
        if i+h == self.bin_height:
            adjacent_direction += 1
        else:
            if sum(self[i+h, j:j+w]) != 0:
                adjacent_direction += 1
        # left
        if j == 0:
            adjacent_direction += 1
        else:
            if sum(self[i:i+h, j-1]) != 0:
                adjacent_direction += 1
        # right
        if j+w == self.bin_width:
            adjacent_direction += 1
        else:
            if sum(self[i:i+h, j+w]) != 0:
                adjacent_direction += 1
        # at least having adjacency in two directions
        return adjacent_direction >= 2

    def get_moves_for_square(self, items_list_board, item_idx):
        # get moves for each available item -xw
        # then get all valid moves in get_legal_moves
        item = items_list_board[item_idx] # board format
        assert sum(sum(item)) > 0
        w = sum(item[0,:])
        h = sum(item[:,0]) 
        moves = []
        # current item to be placed
        for i in range(self.bin_height-h+1):
            for j in range(self.bin_width-w+1):
                if sum(sum(self[i:i+h, j:j+w])) == 0:
                    adjacent = self.get_adjacency(i, j, h, w)
                    if adjacent:
                        moves.append((item_idx, i, j))
        return moves

    def execute_move(self, move, w, h):
        """Only update board (bin).
        """
        # return new board and new items_all
        # move: (i, j)
        # items_list = items_list.copy()
        pieces = self.pieces.copy()
        i, j = move
        assert(sum(sum(pieces[i:i+h, j:j+w])) == 0)
        pieces[i:i+h, j:j+w] = 1
        self.pieces = pieces.copy()

    # @staticmethod
    # def _increment_move(move, direction, n):
    #     # print(move)
    #     """ Generator expression for incrementing moves """
    #     move = list(map(sum, zip(move, direction)))
    #     #move = (move[0]+direction[0], move[1]+direction[1])
    #     while all(map(lambda x: 0 <= x < n, move)): 
    #     #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
    #         yield move
    #         move=list(map(sum,zip(move,direction)))
    #         #move = (move[0]+direction[0],move[1]+direction[1])

    # def init_item_plane(self, w, h):
    #     for i in range(h):
    #         for j in range(w):
    #             self[i][j] = 1

    # def flip_item_plane(self):
    #     for i in range(self.bin_height):
    #         for j in range(self.bin_width):
    #             self[i][j] *= -1