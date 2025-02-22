import random

class Ghost:
    def __init__(self, position):
        self.position = position

    def move_towards(self, target, game_map):
        x, y = self.position
        tx, ty = target
        possible_moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid_moves = [pos for pos in possible_moves if game_map[pos[0]][pos[1]] != 1]

        if valid_moves:
            self.position = min(valid_moves, key=lambda pos: abs(pos[0] - tx) + abs(pos[1] - ty))