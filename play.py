import pygame
import torch
from environment.pacman import PacManEnv
from environment.assets_loader import load_images, load_sounds
from agents.dqn import DQNAgent


def animate_pacman(screen, images, direction, pos, frame, cell_size):
    sprite = images["pacman"][direction][frame % 3]
    sprite = pygame.transform.scale(sprite, (cell_size, cell_size))
    screen.blit(sprite, (pos[0] * cell_size, pos[1] * cell_size))


def draw_ghost(screen, images, ghost_type, pos, cell_size, vulnerable):
    ghost_img = images["ghosts"]["blue_ghost"] if vulnerable else images["ghosts"][ghost_type]
    ghost_img = pygame.transform.scale(ghost_img, (cell_size, cell_size))
    screen.blit(ghost_img, (pos[0] * cell_size, pos[1] * cell_size))


def draw_food(screen, images, maze, cell_size):
    for y, row in enumerate(maze):
        for x, tile in enumerate(row):
            if tile == 2:
                dot_img = pygame.transform.scale(images["foods"]["dot"], (10, 10))
                screen.blit(dot_img, (x * cell_size + cell_size // 2 - 5, y * cell_size + cell_size // 2 - 5))
            elif tile == 3:
                apple_img = pygame.transform.scale(images["foods"]["apple"], (20, 20))
                screen.blit(apple_img, (x * cell_size + cell_size // 2 - 10, y * cell_size + cell_size // 2 - 10))


def main():
    pygame.init()
    pygame.mixer.init()  # Initialize sound mixer
    cell_size = 40  # Larger cell size for better visibility
    grid_size = 20  # Use a 20x20 grid
    screen_width = grid_size * cell_size
    screen_height = grid_size * cell_size + 80  # Extra space for score display
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man RL Edition 🎮 (RL Controlled)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    images = load_images()
    sounds = load_sounds()

    # Show title screen
    sounds["pacman_beginning"].play()
    title_image = images["title"]["pacman_title"]
    scaled_title = pygame.transform.scale(title_image, (screen_width, screen_height - 80))
    screen.blit(scaled_title, (0, 0))
    pygame.display.update()
    pygame.time.wait(3000)

    # Initialize environment and RL agent
    env = PacManEnv(grid_size=grid_size, num_ghosts=4)
    state = env.reset()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n

    agent = DQNAgent(state_dim, action_size)
    try:
        agent.model.load_state_dict(torch.load("dqn_model_final.pth", map_location=torch.device('cpu')))
        print("Loaded trained model weights.")
    except Exception as e:
        print("Could not load trained model weights. Proceeding with untrained agent.")

    # Start background music (ensure the key is "background" in your assets loader)
    sounds["backgroud"].play(-1)

    running = True
    frame = 0

    while running:
        # Render the maze using the environment's render method
        env.render(screen, cell_size)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Use the RL agent to choose an action (greedy, epsilon=0)
        action_index = agent.choose_action(state, epsilon=0)
        state, reward, done, _ = env.step(action_index)

        # Draw ghosts and Pac-Man over the rendered maze
        for idx, ghost in enumerate(env.ghost_positions):
            ghost_keys = list(images["ghosts"].keys())
            ghost_type = ghost_keys[idx % len(ghost_keys)]
            draw_ghost(screen, images, ghost_type, ghost, cell_size, env.power_timer > 0)

        direction_mapping = {0: "up", 1: "down", 2: "left", 3: "right"}
        direction_str = direction_mapping[action_index]
        animate_pacman(screen, images, direction_str, env.pacman_pos, frame, cell_size)

        score_text = font.render(f"Score: {env.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, grid_size * cell_size + 10))

        pygame.display.update()
        frame += 1
        clock.tick(10)

        if done:
            sounds["pacman_death"].play()
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
