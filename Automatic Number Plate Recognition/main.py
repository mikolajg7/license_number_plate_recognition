import cv2
import numpy as np
import imutils
import pytesseract

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Zastosowanie korekcji gamma za pomocą tabeli poszukiwań
    return cv2.LUT(image, table)


img = cv2.imread('license_plates/group3/Cars1.png')

img = adjust_gamma(img, 2.0)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Zastosowanie filtru bilateralnego w celu redukcji szumu
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Detekcja krawędzi za pomocą algorytmu Canny'ego
edged = cv2.Canny(bfilter, 30, 200)

# Wyszukanie konturów na obrazie z detekcją krawędzi
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

# Posortowanie konturów na podstawie pola powierzchni i zachowanie tylko największych
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None

# Iteracja po konturach
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    (x, y, w, h) = cv2.boundingRect(contour)
    ar = w / float(h)
    # Sprawdzenie, czy współczynnik proporcji mieści się w określonym zakresie
    if ar >= 2.5 and ar <= 4.7:
        location = contour
        break


if location is not None:
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = img[x1:x2 + 1, y1:y2 + 1]

    # Wyświetlenie wyciętego fragmentu obrazu
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)

    # OCR
    custom_config = r'--oem 3 --psm 13'
    text = pytesseract.image_to_string(cropped_image, config=custom_config)
    print(text)
else:
    print("Nie znaleziono tablicy rejestracyjnej")
