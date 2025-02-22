"""
Collision detection for Pac-Man game.
"""

import numpy as np

def check_collision(rect1, rect2):
    """
    Check if two rectangles collide.
    rect: (x, y, width, height)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2)

if __name__ == "__main__":
    print("Collision test:", check_collision((0, 0, 50, 50), (25, 25, 50, 50)))
