import pygame
import os

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "assets")

def load_images():
    """
    Loads character sprites (Pac-Man, ghosts, foods, title images).
    """
    return {
        "pacman": {
            d: [pygame.image.load(os.path.join(ASSETS_PATH, f"pacman-{d}", f"{i}.png"))
                for i in range(1, 4)]
            for d in ["up", "down", "left", "right"]
        },
        "ghosts": {
            name: pygame.image.load(os.path.join(ASSETS_PATH, "ghosts", f"{name}.png"))
            for name in ["blinky", "pinky", "inky", "clyde", "blue_ghost"]
        },
        "foods": {
            item: pygame.image.load(os.path.join(ASSETS_PATH, "other", f"{item}.png"))
            for item in ["dot", "apple", "strawberry"]
        },
        "title": {
            os.path.splitext(img)[0]: pygame.image.load(os.path.join(ASSETS_PATH, "title", img))
            for img in ["high_score.jpg", "pacman_title.png"]
        }
    }

def load_sounds():
    """
    Loads sound files from assets/sounds.
    """
    sound_files = [
        "backgroud.mp3",  # Make sure the name matches the actual file
        "pacman_beginning.wav",
        "pacman_chomp.mp3",
        "pacman_death.wav",
        "pacman_eatghost.wav"
    ]

    sounds = {}
    for filename in sound_files:
        sound_path = os.path.join(ASSETS_PATH, "sounds", filename)
        if os.path.exists(sound_path):
            key_name = os.path.splitext(filename)[0]
            sounds[key_name] = pygame.mixer.Sound(sound_path)
        else:
            print(f"⚠️ Warning: Sound file not found: {sound_path}")
    return sounds

def load_tiles():
    """
    Loads tile images for floor, wall, and power pellet.
    """
    tiles = {}

    # ✅ Load floor tile
    floor_path = os.path.join(ASSETS_PATH, "title", "floor.png")
    if os.path.exists(floor_path):
        tiles["floor"] = pygame.image.load(floor_path).convert_alpha()
    else:
        print(f"⚠️ Warning: Tile image not found: {floor_path}")

    # ✅ Load wall tile
    wall_path = os.path.join(ASSETS_PATH, "title", "wall.png")
    if os.path.exists(wall_path):
        tiles["wall"] = pygame.image.load(wall_path).convert_alpha()
    else:
        print(f"⚠️ Warning: Tile image not found: {wall_path}")

    # ✅ Load power pellet tile
    power_pellet_path = os.path.join(ASSETS_PATH, "title", "power_pellet.png")
    if os.path.exists(power_pellet_path):
        tiles["power_pellet"] = pygame.image.load(power_pellet_path).convert_alpha()
    else:
        print(f"⚠️ Warning: Power pellet tile not found: {power_pellet_path}")

    return tiles
