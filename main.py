import cv2
import numpy as np
import imutils
import pytesseract

# Function to adjust the gamma of an image
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Read the image
img = cv2.imread('license_plates/group3/Cars1.png')

# Adjust the gamma of the image
img = adjust_gamma(img, 2.0)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to reduce noise
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection using Canny algorithm
edged = cv2.Canny(bfilter, 30, 200)

# Find contours in the edge detected image
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

# Sort contours based on area and keep only the largest ones
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None

# Iterate over contours
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    (x, y, w, h) = cv2.boundingRect(contour)
    ar = w / float(h)
    if ar >= 2.5 and ar <= 4.7:
        location = contour
        break

# If a contour with the correct aspect ratio was found
if location is not None:
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Find the bounding box of the license plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = img[x1:x2 + 1, y1:y2 + 1]

    # OCR
    custom_config = r'--oem 3 --psm 13'
    text = pytesseract.image_to_string(cropped_image, config=custom_config)
    print("Detected Text:", text)

    # Display the cropped image in a persistent window
    cv2.imshow('Cropped Image', cropped_image)
    print("Press 'q' to close the image window.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
else:
    print("License plate not found")