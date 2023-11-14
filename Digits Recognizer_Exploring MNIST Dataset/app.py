import pygame, sys
import numpy as np
from keras.models import load_model
from pygame.locals import *

import cv2
white = (255,255,255)
red = (255, 0, 0)
predict = True
model = load_model('.\code_dev\Mymodel.h5')
pygame.init() 
boundaryinc = 5
windowsizex = 800
windowsizey = 480
labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

img_save = False
dis_surf = pygame.display.set_mode((800,480))
white_int = dis_surf.map_rgb(white)
pygame.display.set_caption('Handwritten digits reg')

iswriting = False
num_xcord = []
num_ycord = []
rect_min_x, rect_max_x, rect_min_y, rect_max_y = 0, 0, 0, 0  
inmg_cnt = 1

# ...

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(dis_surf, white, (xcord, ycord), 4, 0)
            num_xcord.append(xcord)
            num_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            num_xcord = sorted(num_xcord)
            num_ycord = sorted(num_ycord)

            rect_min_x, rect_max_x = max(num_xcord[0] - boundaryinc, 0), min(windowsizex, num_xcord[-1] - boundaryinc)
            rect_min_y, rect_max_y = max(num_ycord[0] - boundaryinc, 0), min(windowsizey, num_ycord[-1] - boundaryinc)

            num_xcord = []
            num_ycord = []
            img_arr = np.array(pygame.PixelArray(dis_surf), dtype=np.uint8)
            if img_save:
                cv2.imwrite(f'image_{inmg_cnt}.png', img_arr)
                inmg_cnt += 1

            if predict:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / white_int
                predicted_label = str(labels[np.argmax(model.predict(image.reshape(-1, 28, 28, 1)))])
                pygame.draw.rect(dis_surf, red, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 3)
                font = pygame.font.Font(None, 36)
                text = font.render(predicted_label, True, white)
                dis_surf.blit(text, (rect_min_x, rect_min_y))
        if event.type == KEYDOWN:
            if event.unicode == 'N':
                dis_surf.fill((0, 0, 0))  

    pygame.display.update()
