import pygame

def load_image(path, size=(32, 32)):
    image = pygame.image.load(path)
    return pygame.transform.scale(image, size)

def draw_text(surface, text, position, size=24, color=(255, 255, 255)):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)