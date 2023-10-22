import cv2
import os
import numpy as np


def crop_and_preprocess_image(image_path, x1, x2, y1, y2, target_size=(300, 300)):
    """
    This method is used to crop the image so only the mosquito is returned using the bounding box
    from the CSV file as reference, it also resizes the image
    """
    # Load the image from the file
    image = cv2.imread(image_path)
    #print(image.shape)
    #print(" THIS IS THE SHAPE OF THE IMAGE")
    # Check if the loaded image is not empty
    if image is not None and image.size != 0:
        # This attempts to crop the image using the coordinates (x1, x2, y1, y2)
        cropped_image = image[y1:y2, x1:x2]
        #print(cropped_image.shape)
        #print(" THIS IS THE SHAPE OF THE IMAGE cropped")
        # Check if the cropped image is not empty
        if cropped_image is not None and cropped_image.size != 0:
         # Resize the cropped image to the target size
            cropped_image = cv2.resize(cropped_image, target_size)
        return cropped_image


def display_image(cropped_image, class_label="nothing", image_name="nothing"):
    """
    This method is used to display an image
    """
    # Close the image window
    cv2.imshow(class_label, cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
