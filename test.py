import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
     #green mask
    lower_green = np.array( [40,40,40], dtype = "uint8")
    upper_green = np.array( [70,255,255], dtype = "uint8")
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    output = cv2.bitwise_and(image, image, mask=mask_green)
   #gaussian blur
    kernel_size = 3
    gauss =cv2.GaussianBlur(mask_green,(5,5),kernel_size)
     #dilation
    kernel = np.ones((kernel_size*2,kernel_size*2),np.uint8)
    dilation_image = cv2.dilate(mask_green, kernel, iterations=1)
    #morph close
    closing = cv2.morphologyEx(dilation_image, cv2.MORPH_CLOSE, kernel)
     #remove small blobs
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
    #connectedComponentswithStats yields every separated component with information on each of them, such as size
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    min_size = 150  #num pixels

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    cv2.imshow('final',img2)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow('hsv',img_hsv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()