from skimage.metrics import structural_similarity 
import imutils
import cv2
from PIL import Image
import requests

# Open the image and display
# stream=True -> Get the image without save it
# .raw -> Get the image information from internet
original = Image.open(requests.get('https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg', stream=True).raw)
tampered = Image.open(requests.get('https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png', stream=True).raw)

# The file format of the source file
print("Original image format: ", original.format)
print("Tampered image format: ", tampered.format)

print('--------------------------------')

# Image size, in pixels. The size is given as a tuple (width, height)
print("Original image size: ", original.size)
print("Tampered image size: ", tampered.size)

# Resize Image
original = original.resize((280,160))
print("Original size: ", original.size)
original.save("C:/Users/Dule/Desktop/pan_card_tampering/image/original.png") # Save image
tampered = tampered.resize((280,160))
print("Tampered size: ", tampered.size)
tampered.save("C:/Users/Dule/Desktop/pan_card_tampering/image/tampered.png") # Save image

# Display Original Image
print(original.show())

# Display user given Image
print(tampered.show())

# load the two input images
original  = cv2.imread("C:/Users/Dule/Desktop/pan_card_tampering/image_1/original.png")
tampered = cv2.imread("C:/Users/Dule/Desktop/pan_card_tampering/image_1/tampered.png")

# Convert the images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

# Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


# Calculating threshold and contours
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


# loop over the contours
for c in cnts:
    # applaying contours on image
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x,y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(tampered, (x,y), (x + w, y + h), (0, 0, 255), 2)

# Display original image with the contour
print("Original Format Image")
Image.fromarray(original).show()

# Display tampared image with the contour
print("Tampared Format Image")
Image.fromarray(tampered).show()



# ALL THE BLACK IS THE DIFFERENCE -> 100% FAKE 
# Display difference image with black
print("Different Image")
Image.fromarray(diff).show()


# ALL THE WHITE IS THE DIFFERENCE -> 100% FAKE 
# Display difference image with white
print("Threshold Image")
Image.fromarray(thresh).show()