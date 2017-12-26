resources = [
    'wood',
    'sheep',
    'rock',
    'clay',
    'wheat',
    'desert'
]
players = [
    'orange',
    'white',
    'blue'
]
expected_pips = {
    # averages at 58/5
    'rock': (58/18)*3,
    'clay': (58/18)*3,
    'wheat': (58/18)*4,
    'wood': (58/18)*4,
    'sheep': (58/18)*4,
    'desert': 0
}
resource_weighting = {
    'wheat': 1.15,
    'wood': 1,
    'sheep': 0.9,
    'rock': 1,
    'clay': 1,
    'desert': 0
}
resource_colours = {
    'wood': 'darkgreen',
    'sheep': 'lightgreen',
    'rock': 'grey',
    'clay': 'red',
    'wheat': 'orange',
    'desert': None
}
resource_occurances = {
    'rock': 3,
    'clay': 3,
    'wheat': 4,
    'wood': 4,
    'sheep': 4,
    'desert': 1
}

rolls = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]
reslist = []
for res, n in resource_occurances.items():
    reslist += [res]*n

legal_tiles = [
    (2,4),(4,4),(6,4),
    (1,3),(3,3),(5,3),(7,3),
    (0,2),(2,2),(4,2),(6,2),(8,2),
    (1,1),(3,1),(5,1),(7,1),
    (2,0),(4,0),(6,0)
]

legal_verts = ([(i, 5) for i in range(2, 6+3)] +
               [(i, 4) for i in range(1, 7+3)] +
               [(i, 3) for i in range(0, 8+3)] +
               [(i, 2) for i in range(0, 8+3)] +
               [(i, 1) for i in range(1, 7+3)] +
               [(i, 0) for i in range(2, 6+3)])

roll_map = {
    2:1,3:2,4:3,5:4,6:5,
    8:5,9:4,10:3,11:2,12:1
}
