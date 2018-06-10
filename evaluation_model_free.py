from __future__ import division
from itertools import cycle
from model_free import Model_free
import random
import sys
import pygame
from pygame.locals import *
import time
import json
import numpy as np

sys.path.append("..")
sys.path.append("game")
import flappy_bird_utils

FPS = 60
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


def mainGame(mf):
    pygame.event.pump()

    score = playerIndex = loopIter = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    playerx, playery = int(SCREENWIDTH * 0.2), int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
    basex = 0
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    start_time = time.time()
    while True:
        # if -playerx + lowerPipes[0]['x'] > -30:
        #     myPipe = lowerPipes[0]
        # else:
        #     myPipe = lowerPipes[1]

        if -playerx + lowerPipes[0]['x'] > -30:
            myPipe = lowerPipes[0]
            nextPipe = lowerPipes[1]
        else:
            myPipe = lowerPipes[1]
            if len(lowerPipes) > 2:
                nextPipe = lowerPipes[2]
            else:
                nextPipe = {'x': 0, 'y': 0}


        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
                    # SOUNDS['wing'].play()


        state = mf.map_state(-playerx + myPipe['x'], - playery + myPipe['y'], playerVelY, -playerx + nextPipe['x'], - playery + nextPipe['y'])

        if not mf.qvalues.has_key(state):
            action = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        else:
            action = np.argmax(mf.qvalues[state])

        if action:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
                # SOUNDS['wing'].play()

        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0]:
            return score

        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                # SOUNDS['point'].play()

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)


        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX


        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))


        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def getRandomPipe():
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']

    # if player crashes into ground
    if (player['y'] + PLAYER_HEIGHT >= BASEY - 1 ) or (player['y'] + PLAYER_HEIGHT <= 0):
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'], PLAYER_WIDTH, PLAYER_HEIGHT)

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]



def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False


def getHitmask(image):
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask



def main():
    # Initialize the bot
    model_free = Model_free()
    num = 0
    size = 100
    while True:
        num += 1000
        total = 0
        model_free.qvalues = {}
        try:
            file_path = model_free.addr + '/qvalues' + str(num) + '.json'
            print file_path
            f = open(file_path, 'r')
        except IOError:
            print "Read file error!"
            break
        else:
            model_free.qvalues = json.load(f)
            f.close()
        for j in range(size):
            total += mainGame(model_free)

        f_time = open(model_free.addr + '/data/evaluations.txt', 'a')
        f_time.write(str(total/size) + " ")
        print total
        f_time.close()


def evaluation():
    model_free = Model_free()

    try:
        file_path = model_free.addr + '/qvalues10000.json'
        print file_path
        f = open(file_path, 'r')
    except IOError:
        print "Read file error!"
    else:
        model_free.qvalues = json.load(f)
        f.close()

    while True:
        mainGame(model_free)


if __name__ == '__main__':
    main()

