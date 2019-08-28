import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from catan.items import Tile, Tiles, Vertex, Verts
from catan.constants import *


def random_tiles():
    """Create a number list of tiles by shuffling legal tiles
    """
    tiles = []
    random.shuffle(legal_tiles)
    for i, (x, y) in enumerate(legal_tiles):
        res = reslist[i]
        if res == 'desert':
            # doesn't get a roll marker
            roll = None
        else:
            roll = rolls[i]
        tiles.append(Tile(x, y, resource=res, roll=roll))

    return tiles


class Board:
    """A container for vertices and tiles

    Contains board state (what has been settled) and functions to calculate worth of a vertex.

    a standard board has...
        19 tiles
            3 rock
            3 clay
            4 sheep
            4 wood
            4 wheat
            1 desert
        54 vertices
    """
    def __init__(self, tiles):
        """ tiles is a list of legal tiles
        """
        # automatically handle if is already Tiles type
        self.tiles = Tiles(tiles)
        # initialise with an empty set of vertices
        self.verts = Verts([Vertex(x, y) for x, y in legal_verts])
        self.total_pips = {res: sum(tile.pips for tile in self.tiles.resource(res)) for res in resources}

    @classmethod
    def random(cls):
        """Randomly create a board
           Do not allow a 6 and 8 to be adjacent
        """
        is_bad = True
        while is_bad:
            brd = cls(random_tiles())
            is_bad = brd._six_eight_adjacent()
        return brd

    def _six_eight_adjacent(self):
        """Are any 6 or 8 tiles next to each other?
        """
        for tile in self.tiles.roll(6):
            for t2 in self.tt(tile.x, tile.y):
                if t2.roll == 8 or t2.roll == 6:
                    return True
        for tile in self.tiles.roll(8):
            for t2 in self.tt(tile.x, tile.y):
                if t2.roll == 8 or t2.roll == 6:
                    return True

        return False

    def resource_table(self):
        """A DataFrame summarising the board
        """
        table = pd.DataFrame({'expected_pips': expected_pips,
                              'total_pips': self.total_pips})
        table['more_than_expec'] = table['total_pips'] / table['expected_pips']
        return table

    # actions
    def settle(self, x, y, player):
        self.verts[x, y].settle(player)
        for vert in self.vv(x, y):
            vert.block()

    def unsettle(self, x, y):
        # unsettle and unblock
        self.verts[x, y].unsettle()
        for vert in self.vv(x, y):
            vert.unblock()

        # re-settle all vertices in order to check blocking is correct
        for vert in self.verts:
            if vert.settled:
                self.settle(vert.x, vert.y, vert.settled)


    # get adjacent items
    def tt(self, x, y):
        adj = [
            self.tiles[x+2, y],
            self.tiles[x-2, y],
            self.tiles[x-1, y+1],
            self.tiles[x+1, y+1],
            self.tiles[x-1, y-1],
            self.tiles[x+1, y-1]
        ]
        return Tiles(adj)

    def tv(self, x, y):
        # always 6
        adj = [
            self.verts[x, y],
            self.verts[x+1, y],
            self.verts[x+2, y],
            self.verts[x, y+1],
            self.verts[x+1, y+1],
            self.verts[x+2, y+1]
        ]
        return Verts(adj)

    def vv(self, x, y):
        # if sum is even, go up
        # if sum is odd, go down
        if (x+y)%2 == 0:
            adj = [
                self.verts[x, y+1],
                self.verts[x-1, y],
                self.verts[x+1, y]
            ]
        else:
            adj = [
                self.verts[x, y-1],
                self.verts[x-1, y],
                self.verts[x+1, y]
            ]
        return Verts(adj)

    def vt(self, x, y):
        if (x+y)%2 == 0:
            adj = [
                self.tiles[x, y],
                self.tiles[x-2, y],
                self.tiles[x-1, y-1]
            ]
        else:
            adj = [
                self.tiles[x-1, y],
                self.tiles[x, y-1],
                self.tiles[x-2, y-1]
            ]
        return Tiles(adj)


    # metrics- return the value for a particular vertex
    metrics = ['pips', 'relpips', 'pipworth', 'ave_potential', 'blocking']

    def pips(self, x, y, *_):
        """Total adjacent pips to vertex
        """
        return sum(tile.pips for tile in self.vt(x, y))

    def relpips(self, x, y, *_):
        """Total worth of adjacent resources to vertex, considering resource scarcity and
           exogenous weights
        """
        worth = 0
        for tile in self.vt(x, y):
            res = tile.resource
            if res == 'desert':
                continue
            worth += tile.pips*resource_weighting[res]/self.total_pips[res]
        # scale worth to be on the same scale as regular pips. this is the average expected pips
        return worth*(58/5)

    def pipworth(self, x, y, player=None):
        """Total worth of adjacent resources to vertex, considering resource scarcity,
           exogenous weights and players current settlements
        """
        # a mapping from resource type to a list of worths
        pipmap = defaultdict(list)

        for tile in self.vt(x, y):
            res = tile.resource
            if res == 'desert':
                continue
            pipmap[res].append(tile.pips*resource_weighting[res]/self.total_pips[res])

        # Player state adjustment
        # add in a 0 pip placeholder for all already settled spots- lets us decay all subsequent resources
        # there is quite a bit of duplication here, because, for a certain player this code
        # will get repeated a lot. solution is maybe to pass a pipmap in and calculate under best()?
        for vert in self.verts.player(player):
            for t in self.vt(x, y):
                pipmap[t.resource].append(0)

        # decay the higher value or later resources by more
        worth = 0
        for res, piplist in pipmap.items():
            worth += np.dot(sorted(piplist), resdecay[:len(piplist)])

        # roll diversity??
        return worth*(58/5)

    def ave_potential(self, x, y, player=None):
        """The maximum average potential score, considering the (up to 3) roads a player may
           build from a settlement
        """
        return max(self.all_potentials(x, y, player).values())

    def all_potentials(self, x, y, player=None):
        """Considers all potential settlement locations
           returns a dictionary of x,y: pips, representing
           up to 3 vertex locations 1 step away from x,y.
           pips is the sum of the locations one road away from this vertex
        """
        vert_worth = {}
        for vert in self.vv(x, y):
            worth = 0
            for vert2 in self.vv(vert.x, vert.y):
                if (vert2.x, vert2.y) == (x, y):
                    continue
                worth += self.pipworth(vert2.x, vert2.y, player)
            vert_worth[vert.x, vert.y] = worth/2
        return vert_worth

    def blocking(self, x, y, player=None):
        """Blocking score- the average of the pipworth of the 3 surrounding vertices,
           which may not be settled if a settlement if placed here

        TODO: this should maybe be considered as blocking other's pairs- eg. a player really needs rock, so this is a good block
        """
        pips = 0
        for vert in self.vv(x, y).nonblocked():
            pips += self.pipworth(vert.x, vert.y, player)
        return pips/3


    # get all metrics
    def worths(self, method='pips', player=None):
        """Given a method, calculate the worth of every nonblocked vertex
        """
        worths = {}
        for vert in self.verts.nonblocked():
            x, y = vert.x, vert.y
            worths[x, y] = getattr(self, method)(x, y, player)
        return worths

    def best(self, sort_method='relpips', player=None):
        """Return a sorted dataframe of vertex worths (for a particular player)
        """
        df = pd.DataFrame({m: self.worths(m, player) for m in self.metrics}).rename_axis(['x', 'y'])
        df['total'] = df.sum('columns')
        df['tiles'] = df.index.map(lambda pos: self.vt(*pos).short_print())
        return df.sort_values(sort_method, ascending=False)

    def best_pair(self, method='relpips', player=None):
        """The best two positions for a player at this point in time
           This will implicitly take into account the fact that two of the same resource is worth less
        """
        pairs = {}
        ranked = self.best(player=player)
        for (x, y), first_vert_worth in ranked.iterrows():
            # fake settling this location and find the other best location
            self.settle(x, y, player=player)
            best_pair = self.best(player=player).reset_index().iloc[0]
            pairs[x, y, best_pair['x'], best_pair['y']] = first_vert_worth[method] + best_pair[method]
            self.unsettle(x, y)

        df = pd.DataFrame({method: pairs}).rename_axis(['x1', 'y1', 'x2', 'y2'])
        return df.sort_values(method, ascending=False)


    # def b2(self, method='relpips', player=None, player_c):
    #     """Return a dataframe of best positions for the second to last placement (B2 in a A1 B1 C1 C2 B2 A2 scheme)
    #     """
    #     worths = {}
    #     for vert in self.verts.nonblocked():
    #         x, y = vert.x, vert.y
    #         # pretend to settle, then simulate next players actions
    #         self.settle(x, y, player=player)
    #         # NEST HERE for next players action afterwards:
    #         # for vert in self.verts.nonblocked():
    #         best_pair = self.best(player=player_c).reset_index().iloc[0]  # TODO we don't need to calculate for every method, which is faster
    #         self.settle(best_pair[x], best_pair[y], player_c)  # TODO we actually don't need this to block, which is faster to unsettle again

    #         # get the worth of this position, knowing the next players action
    #         self.unsettle(x, y)
    #         worths[x, y] = getattr(self, method)(x, y, player)  # self.pipworth(x, y, player=player)
            
    #         # unsettle next players best action
    #         self.unsettle(best_pair[x], best_pair[y])
        
    #     df = pd.Series(worths)
    #     df.index.names = ['x', 'y']
    #     return df.to_frame(method).sort_values(method, ascending=False)


    # plotting
    def plot(self):
        fig, ax = plt.subplots(figsize=(10,10))
        plt.axis('off')
        # Ocean
        # octagon = ([0.7, -0.3], [4.2, -0.3], [5.5, 2.5], [4.2, 5.2], [0.7, 5.2], [-0.5, 2.5])
        # ax.add_patch(patches.Polygon(octagon, color='blue'))

        self.tiles.plot(ax)
        self.verts.plot(ax)
    

    def plot_best(self, method='total', n=5):
        for i, ((x, y), row) in enumerate(self.best(method).head(n).iterrows(), start=1):
            x, y = Vertex(x, y).real_xy
            plt.plot(x, y, marker='x', color='purple', markeredgewidth=3)
            plt.text(x+0.1, y, i, color='purple', weight='bold')
