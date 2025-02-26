import json
import os
import pygame

LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    """Load leaderboard data from JSON file."""
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as file:
            return json.load(file)
    return []

def save_score(player_name, score):
    """Save new score and update leaderboard."""
    leaderboard = load_leaderboard()
    leaderboard.append({"name": player_name, "score": score})
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    leaderboard = leaderboard[:10]  # Keep top 10 scores

    with open(LEADERBOARD_FILE, "w") as file:
        json.dump(leaderboard, file, indent=4)
    print(f"[Leaderboard] Updated with {player_name}: {score}")

def display_leaderboard(screen, font, cell_size):
    """Display leaderboard on the Pygame screen."""
    leaderboard = load_leaderboard()
    screen.fill((0, 0, 0))
    title = font.render("🏆 Leaderboard 🏆", True, (255, 255, 0))
    screen.blit(title, (cell_size * 3, 20))

    y_offset = 70
    for idx, entry in enumerate(leaderboard, 1):
        entry_text = font.render(f"{idx}. {entry['name']} - {entry['score']}", True, (255, 255, 255))
        screen.blit(entry_text, (cell_size * 2, y_offset))
        y_offset += 30

    pygame.display.update()
