import random

from catan.constants import players
from catan.board import Board
from catan.items import Verts


class Player:
    def __init__(self,  colour, starting):
        if colour not in players:
            raise ValueError('illegal colour {}'.format(colour))
        self.colour = colour
        self.starting = starting
        self.verts = Verts([])
        self.cards = []
        self.rescards = []

    def settle(self, vertex):
        self.verts.append(vertex)

    def pickup(self, card):
        self.cards.append(card)


class Game:
    """ holds a board and players
    """
    def __init__(self, tiles, colours):
        self.players = {}
        for i, colour in colours:
            self.players[colour] = Player(colour, i)

        self.current_player = self.players[colours[0]]
        self.board = Board(tiles)

    def set_current_player(self, colour):
        self.current_player = self.players[colour]

    def roll(self):
        rolled = random.randint(1, 6) + random.randint(1, 6)
        for tile in self.board.tiles.roll(rolled):
            for vert in self.board.tv(tile.x, tile.y):
                if vert.settled:
                    players[vert.settled].pickup(tile.resource)
        # next player?
