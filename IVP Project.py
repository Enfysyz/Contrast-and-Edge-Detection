# import the necessary packages
from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import argparse
import imutils
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                default="examples", help="path to input directory of images")
ap.add_argument("-t", "--thresh", type=float, default=0.35,
                help="threshold for low contrast")
args = vars(ap.parse_args())

# grab the paths to the input images
imagePaths = sorted(list(list_images(args["input"])))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image from disk
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    text = "Low contrast: No"
    color = (0, 255, 0)

    # check to see if the image is low contrast
    if is_low_contrast(gray, fraction_threshold=args["thresh"]):
        # update the text and color
        text = "Low contrast: Yes"
        color = (0, 0, 255)

        # converting to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # blur the image slightly and perform edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        blurred_enhanced = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
        edged_enhanced = cv2.Canny(blurred_enhanced, 30, 150)

        image = np.hstack((image, enhanced_img))
        edged = np.hstack((edged, edged_enhanced))

    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

    # draw the text on the output image
    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2)
    # show the output image and edge map
    cv2.imshow("Image", image)
    cv2.imshow("Edge", edged)
    cv2.waitKey(0)
