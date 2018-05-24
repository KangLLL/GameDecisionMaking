import pygame
import sys
import numpy as np

sys.path.append("game/")
import wrapped_flappy_bird as game

if __name__ == "__main__":
    game_state = game.GameState()
    R = 0
    while(1):
        action = np.zeros(2)
        action[0] = 1
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[0] = 0
                    action[1] = 1
        _, r, terminal = game_state.frame_step(action)
        if r == 1:
            R += 1
        if terminal:
            print("You got %d points"%(R))
            R = 0
