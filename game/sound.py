import pygame

class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {
            'chomp': pygame.mixer.Sound("assets/sounds/pacman_chomp.wav"),
            'death': pygame.mixer.Sound("assets/sounds/pacman_death.wav"),
            'beginning': pygame.mixer.Sound("assets/sounds/pacman_beginning.wav"),
        }

    def play(self, sound_name):
        if sound_name in self.sounds:
            self.sounds[sound_name].play()

if __name__ == "__main__":
    sm = SoundManager()
    sm.play('chomp')
