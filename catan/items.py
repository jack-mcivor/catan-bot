import matplotlib.pyplot as plt
import matplotlib.patches as patches

from catan.constants import legal_tiles, legal_verts, roll_to_pips, resource_colours
import math


def hexcoords_to_real_coords(x, y):
    realx = math.sqrt(3)/2 * x
    realy = 3/2 * y
    return realx, realy


def pip_positions(n):
    if n == 1:
        return [0]
    elif n == 2:
        return [-1, 1]
    elif n == 3:
        return [-2, 0, 2]
    elif n == 4:
        return [-3, -1, 1, 3]
    elif n == 5:
        return [-4, -2, 0, 2, 4]


def roll_to_pips(roll):
    """Map from a roll 1-12 to the number of 'pips' 1-5.
       None will map to 0 pips (desert)
    """
    if roll is None:
        return 0
    if roll == 7 or roll < 2 or roll > 13:
        raise ValueError('illegal roll')
    return roll_map[roll]


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
    
    @property
    def real_centre_xy(self):
        return hexcoords_to_real_coords(self.x+1, self.y+0.5)
    
    def plot(self, ax):
        x, y = self.real_centre_xy
        # plot hexagon
        c = resource_colours[self.resource]
        ax.add_patch(patches.RegularPolygon((x, y), 6, 1, facecolor=c, linewidth=6, edgecolor='floralwhite'))

        if self.resource == 'desert':
            return

        # plot marker
        ax.add_patch(patches.Circle((x, y), radius=0.35, color='papayawhip'))

        # plot roll
        c = 'firebrick' if self.roll == 6 or self.roll == 8 else 'black'
        plt.text(x, y+0.05, self.roll, ha='center', va='center', c=c, fontsize=14, weight='bold')

        # plot pips
        p = [x+i/20 for i in pip_positions(self.pips)]
        plt.plot(p, [y-0.15]*self.pips, marker='.', linestyle='', c=c)


class Tiles:
    """Basically just a dictionary that holds tiles

    Gives us the ability to do a nice [] getitem, and define what happens when
    the item doesn't exist
    """
    def __init__(self, tiles):
        self.map = {(t.x, t.y): t for t in tiles if t is not None}

    def __getitem__(self, pos):
        # x, y = pos
        return self.map.get(pos, None)

    def __len__(self):
        return len(self.map)

    def __repr__(self):
        to_print = 'Tiles({}):'.format(len(self))
        for tile in self:
            to_print += '\n  {}'.format(repr(tile))
        return to_print

    def short_print(self):
        return ' '.join(['{} {}'.format(tile.resource, tile.roll) for tile in self])

    def __iter__(self):
        for (x,y), tile in self.map.items():
            yield tile

    def roll(self, roll):
        for tile in self:
            if tile.roll == roll:
                yield tile

    def resource(self, resource):
        for tile in self:
            if tile.resource == resource:
                yield tile

    def plot(self, ax):
        for tile in self:
            tile.plot(ax)


class Vertex:
    def __init__(self, x, y, port=None):
        if (x, y) not in legal_verts:
            raise ValueError('{}, {} is an illegal position'.format(x, y))
        self.x = x
        self.y = y
        self.settled = False
        self.citied = False
        self.blocked = False
        self.port = port  # typically only certain vertices have ports

    def __repr__(self):
        return 'Vertex(@({},{}))'.format(self.x, self.y)

    @property
    def real_xy(self):
        x, y = self.x, self.y
        offset = 1 - math.sqrt(3)/2
        if (x + y) % 2 == 0:
            y += offset
        else:
            y -= offset

        return hexcoords_to_real_coords(x, y)

    def settle(self, player):
        if self.blocked and not self.settled:
            raise ValueError('vertex is blocked!')
        self.settled = player
        self.blocked = True

    def unsettle(self):
        # have to actually check it's not blocked by another vertex
        self.settled = False
        self.blocked = False

    def city(self, player):
        self.citied = player

    def block(self):
        self.blocked = True

    def unblock(self):
        self.blocked = False

    def plot(self, ax):
        x, y = self.real_xy
        if self.settled:
            ax.add_patch(patches.Rectangle((x-0.13, y-0.1), 0.26, 0.2,
                                        color=self.settled, alpha=1))

        if self.port:
            plt.plot(x, y, marker='.', colour='blue')

class Verts:
    def __init__(self, verts):
        self.map = {(v.x, v.y): v for v in verts if v is not None}

    def __getitem__(self, pos):
        # x, y = pos
        return self.map.get(pos, None)

    def __len__(self):
        return len(self.map)

    def __repr__(self):
        to_print = 'Verts({}):'.format(len(self))
        for vert in self:
            to_print += '\n  {}'.format(repr(vert))
        return to_print

    def __iter__(self):
        for (x,y), vert in self.map.items():
            yield vert

    def nonblocked(self):
        for vert in self:
            if not vert.blocked:
                yield vert

    def player(self, colour):
        for vert in self:
            if vert.settled == colour:
                yield vert

    def append(self, vert):
        self.map[vert.x, vert.y] = vert

    def plot(self, ax):
        for vert in self:
            vert.plot(ax)


class Port:
    types = ['wheat 2:1', 'rock 2:1', 'sheep 2:1', 'clay 2:1', 'wood 2:1', 'any 3:1']
    def __init__(self, port_type):
        if port_type not in types:
            raise ValueError(f'Port type {port_type} not allowed! Must be one of {types}')
        self.type = port_type
