import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from catan.constants import *


def roll_to_pips(roll):
    if roll is None:
        return 0
    if roll == 7 or roll < 2 or roll > 13:
        raise ValueError('illegal roll')
    return roll_map[roll]


def random_tiles():
    tiles = []
    random.shuffle(legal_tiles)
    for i, (x, y) in enumerate(legal_tiles):
        res = reslist[i]
        if res == 'desert':
            roll = None
        else:
            roll = rolls[i]
        tiles.append(Tile(x, y, resource=res, roll=roll))

    return tiles


class Tile:
    def __init__(self, x, y, resource=None, roll=None):
        if (x, y) not in legal_tiles:
            raise ValueError('{}, {} is an illegal position'.format(x, y))
        self.x = x
        self.y = y
        self.resource = resource
        self.roll = roll
        self.pips = roll_to_pips(roll)

    def __repr__(self):
        return 'Tile(@({},{}) {} {})'.format(self.x, self.y, self.resource, self.roll)


class Tiles:
    """Basically just a dictionary that holds tiles

    Gives us the ability to do a nice [] getitem, and define what happens when
    the item doesn't exist
    """
    def __init__(self, tiles):
        self.tiles = {(t.x, t.y): t for t in tiles if t is not None}

    def __getitem__(self, pos):
        # x, y = pos
        return self.tiles.get(pos, None)

    def __len__(self):
        return len(self.tiles)

    def __repr__(self):
        to_print = 'Tiles({}):'.format(len(self))
        for tile in self:
            to_print += '\n  {}'.format(repr(tile))
        return to_print

    def short_print(self):
        return ' '.join(['{} {}'.format(tile.resource, tile.roll) for tile in self])

    def __iter__(self):
        for (x,y), tile in self.tiles.items():
            yield tile

    def roll(self, roll):
        for (x,y), tile in self.tiles.items():
            if tile.roll == roll:
                yield tile


class Vertex:
    def __init__(self, x, y, port=None):
        if (x, y) not in legal_verts:
            raise ValueError('{}, {} is an illegal position'.format(x, y))
        self.x = x
        self.y = y
        self.settled = False
        self.citied = False
        self.blocked = False

    def __repr__(self):
        return 'Vertex(@({},{}))'.format(self.x, self.y)

    def settle(self, player):
        self.settled = player
        self.blocked = True

    def city(self, player):
        self.citied = player

    def block(self):
        self.blocked = True


class Verts:
    def __init__(self, verts):
        # maybe call this map?
        # so we'd have brd.verts.map.items():
        self.verts = {(v.x, v.y): v for v in verts if v is not None}

    def __getitem__(self, pos):
        # x, y = pos
        return self.verts.get(pos, None)

    def __len__(self):
        return len(self.verts)

    def __repr__(self):
        to_print = 'Verts({}):'.format(len(self))
        for vert in self:
            to_print += '\n  {}'.format(repr(vert))
        return to_print

    def __iter__(self):
        for (x,y), vert in self.verts.items():
            yield vert

    def nonblocked(self):
        for (x,y), vert in self.verts.items():
            if not vert.blocked:
                yield vert


class Board:
    """ a container for vertices and tiles

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
        if isinstance(tiles, Tiles):
            self.tiles = tiles
        else:
            self.tiles = Tiles(tiles)

        # initialise with an empty set of vertices
        self.verts = Verts([Vertex(x, y) for x, y in legal_verts])
        self.total_pips = {res: self._pips_by_resource(res) for res in resources}
        self.res_table = self._resource_table()

    @classmethod
    def random(cls):
        is_bad = True
        while is_bad:
            brd = cls(random_tiles())
            is_bad = brd._six_eight_adjacent()
        return brd

    def _six_eight_adjacent(self):
        for tile in self.tiles.roll(6):
            for t2 in self.tt(tile.x, tile.y):
                if t2.roll == 8 or t2.roll == 6:
                    return True
        for tile in self.tiles.roll(8):
            for t2 in self.tt(tile.x, tile.y):
                if t2.roll == 8 or t2.roll == 6:
                    return True

        return False

    def _resource_table(self):
        table = pd.concat([
            pd.Series(expected_pips, name='expected_pips'),
            pd.Series(self.total_pips, name='total_pips')
        ], axis=1)
        table['more_than_expec'] = table['total_pips'] / table['expected_pips']
        return table

    def _pips_by_resource(self, resource):
        """ search total pips by resource
        """
        pips = 0
        for tile in self.tiles:
            if tile.resource == resource:
                pips += tile.pips
        return pips

    # actions
    def settle(self, x, y, player):
        self.verts[x, y].settle(player)
        for vert in self.vv(x, y):
            vert.block()


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
    metrics = ['pips', 'relpips', 'relpips2', 'ave_potential', 'blocking']

    def pips(self, x, y):
        pips = 0
        for tile in self.vt(x, y):
            pips += tile.pips
        return pips

    def relpips(self, x, y):
        """ gets the worth of a resource, considering adjacent tiles
        """
        pips = 0
        for tile in self.vt(x, y):
            res = tile.resource
            if res == 'desert':
                return 0
            pips += tile.pips*resource_weighting[res]/self.total_pips[res]
        return pips*(58/5)

    resdecay = [1, 0.75, 5, 0.25]
    def relpips2(self, x, y):
        """ gets the worth of a resource, considering adjacent tiles
        """
        pipsmap = defaultdict(list)

        for tile in self.vt(x, y):
            res = tile.resource
            if res == 'desert':
                return 0
            pipsmap[res].append(tile.pips*resource_weighting[res]/self.total_pips[res])

        # resource diversity
        pips = 0
        for res, piplist in pipsmap.items():
            pips += np.dot(sorted(piplist)[::-1], self.resdecay[:len(piplist)])
        
        # roll diversity??
        return pips*(58/5)

    def ave_potential(self, x, y):
        return max(self.all_potentials(x, y).values())

    def all_potentials(self, x, y):
        """ considers all potential settlement locations
        returns a dictionary of x,y: pips, representing
        up to 3 vertex locations 1 step away from x,y.
        pips is the sum of the locations one road away from this vertex
        """
        vert_pips = {}
        for vert in self.vv(x, y):
            pips = 0
            for vert2 in self.vv(vert.x, vert.y):
                if (vert2.x, vert2.y) != (x, y):
                    pips += self.relpips(vert2.x, vert2.y)
            vert_pips[vert.x, vert.y] = pips/2
        return vert_pips

    def blocking(self, x, y):
        pips = 0
        for vert in self.vv(x, y):
            pips += self.relpips(vert.x, vert.y)
        return pips/3


    # get all metrics
    def worths(self, method='pips'):
        worths = {}
        for vert in self.verts.nonblocked():
            x, y = vert.x, vert.y
            worths[x, y] = getattr(self, method)(x, y)
        return worths

    def best(self, method='relpips'):
        df = pd.concat([pd.Series(self.worths(m), name=m) for m in self.metrics], axis=1)
        df.index.names = ['x', 'y']
        df['total'] = df.sum('columns')
        df['tiles'] = df.index.map(lambda pos: self.vt(*pos).short_print())
        return df.sort_values(method, ascending=False)


    # plotting
    def plot(self):
        fig, ax = plt.subplots(figsize=(10,10))

        # plot vertices
        for v in self.verts:
            x, y = v.x, v.y
            if (x+y) % 2 == 0:
                y += 0.13
            else:
                y -= 0.13
            plt.plot(x/2, y, marker='.', c='k')
            if v.settled:
                ax.add_patch(patches.Rectangle((x/2-0.1, y-0.1), 0.2, 0.2,
                                               color=v.settled, alpha=0.7))

        # plot tile
        for t in self.tiles:
            # center our tiles
            x, y = t.x+1, t.y+0.5
            if t.roll == 6 or t.roll == 8:
                c = 'r'
            else:
                c = 'k'

            # plot roll
            plt.text(x/2, y, t.roll,
                     horizontalalignment='center',
                     verticalalignment='center', color=c)

            # plot hexagon
            c = resource_colours[t.resource]
            ax.add_patch(patches.RegularPolygon((x/2, y), 6, 0.55,
                                                color=c, alpha=0.5))

    def plot_best(self, method='total', n=5):
        i = 0
        for (x, y), row in self.best(method).head(n).iterrows():
            if (x+y) % 2 == 0:
                y += 0.13
            else:
                y -= 0.13
            i += 1
            plt.plot(x/2, y, marker='o', color='purple')
            plt.text(x/2+0.1, y, i, color='purple')
