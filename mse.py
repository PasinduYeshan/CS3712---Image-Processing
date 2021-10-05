import numpy as np
import cv2


sourceImage = cv2.imread('source.jpg')
processedImage = cv2.imread('processed.jpg')
M, N, C = sourceImage.size

value = 0
for x in range(1,M):
  for y in range(1, N):
    value += (sourceImage[x][y] - processedImage[x][y])**2
  
return value/(M*N)
