from catan.constants import legal_tiles, legal_verts, roll_map


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
