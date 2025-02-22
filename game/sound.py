import pygame

class Sound:
    def __init__(self):
        pygame.mixer.init()
        self.eat_sound = pygame.mixer.Sound("assets/sounds/eat.wav")
        self.death_sound = pygame.mixer.Sound("assets/sounds/death.wav")

    def play_eat(self):
        self.eat_sound.play()

    def play_death(self):
        self.death_sound.play()