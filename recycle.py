import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def drawMatches(img1, kp1, img2, kp2, matches):
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]
	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
	# Place the first image to the left
	out[:rows1,:cols1] = np.dstack([img1, img1, img1])
	out[:rows2,cols1:] = np.dstack([img2, img2, img2])
	for mat in matches:

		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		# x - columns
		# y - rows
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		# Draw a small circle at both co-ordinates
																																        # radius 4
		# colour blue
		# thickness = 1
		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		# Draw a line in between the two points
		# thickness = 1
		# colour blue
		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

	cv2.imshow('Matched Features', out)
	cv2.waitKey(0)
	cv2.destroyWindow('Matched Features')

def getOutline(img):
	newImg = np.ndarray(np.shape(img))
	for r in range(np.shape(img)[0]):
		c = 0
		while(c < np.shape(img)[1]):
			if(img[r][c] > 0):
				newImg[r][c] = 255
				break
			c += 1
	for r in range(np.shape(img)[0]):
		c = np.shape(img)[1] - 1
		while(c >= 0):
			if(img[r][c] > 0):
				newImg[r][c] = 255
				break
			c -= 1
	for c in range(np.shape(img)[1]):
		r = 0
		while(r < np.shape(img)[0]):
			if(img[r][c] > 0):
				newImg[r][c] = 255
				break
			r += 1
	for c in range(np.shape(img)[1]):
		r = np.shape(img)[0] - 1
		while(r >= 0):
			if(img[r][c] > 0):
				newImg[r][c] = 255
				break
			r -= 1
	return newImg

img1 = cv2.imread('milkBlue.png',cv2.IMREAD_GRAYSCALE)
edges1 = cv2.Canny(img1, 100, 200)
out1 = getOutline(edges1)
cv2.imshow('outline', out1)
cv2.waitKey(0)
cv2.destroyWindow('outline')

img2 = cv2.imread('milkRed.png', cv2.IMREAD_GRAYSCALE)
edges2 = cv2.Canny(img2, 100, 200)
out2 = getOutline(edges2)
cv2.imshow('outline', out2)
cv2.waitKey(0)
cv2.destroyWindow('outline')
orb = cv2.ORB()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print 'Past detect/compute'

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

drawMatches(out1, kp1, out2, kp2, matches[:10])
#plt.imshow(img3)
#plt.show()
