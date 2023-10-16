import cv2
import os
import numpy as np


def crop_and_preprocess_image(image_path, x1, x2, y1, y2, target_size=(300, 300)):
    """
    This method is used to crop the image so only the mosquito is returned using the bounding box
    from the CSV file as reference, it also resizes the image, grayscales, and normalizes it before
    returning it.
    """
    # Load the image from the file
    image = cv2.imread(image_path)
    #print(image.shape)
    #print(" THIS IS THE SHAPE OF THE IMAGE")
    # Check if the loaded image is not empty
    if image is not None and image.size != 0:
        # Attempt to crop the image using the coordinates (x1, x2, y1, y2)
        cropped_image = image[y1:y2, x1:x2]
        #print(cropped_image.shape)
        #print(" THIS IS THE SHAPE OF THE IMAGE cropped")
        # Check if the cropped image is not empty
        if cropped_image is not None and cropped_image.size != 0:
         # Resize the cropped image to the desired dimensions
            cropped_image = cv2.resize(cropped_image, target_size)
        return cropped_image

# # This is the original method
# def crop_and_preprocess_image(image_path, x1, x2, y1, y2, target_size=(224, 224)):
#     """
#     This method is used to crop the image so only the mosquito is returned using the bounding box
#     from the CSV file as reference, it also resizes the image, grayscales, and normalizes it before
#     returning it.
#     """
#     # Load the image from the file
#     image = cv2.imread(image_path)
#     print(image.shape)
#     print(" THIS IS THE SHAPE OF THE IMAGE")
#     # Check if the loaded image is not empty
#     if image is not None and image.size != 0:
#         # Attempt to crop the image using the coordinates (x1, x2, y1, y2)
#         cropped_image = image[y1:y2, x1:x2]
#         print(cropped_image.shape)
#         print(" THIS IS THE SHAPE OF THE IMAGE cropped")
#         # Check if the cropped image is not empty
#         if cropped_image is not None and cropped_image.size != 0:
#             # Resize the cropped image to the desired dimensions
#             cropped_image = cv2.resize(cropped_image, target_size)
#             print(cropped_image.shape)
#             print(" THIS IS THE SHAPE OF THE cropped IMAGE resized")
#             # Convert the cropped image to grayscale
#             # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#
#             # Convert the image to float32
#             cropped_image = cropped_image.astype(np.float32)
#             print(cropped_image.shape)
#             print(" THIS IS THE SHAPE OF THE IMAGE as float 32")
#             # Normalize the pixel values to [0, 1]
#             cropped_image /= 255.0
#             print(cropped_image.shape)
#             print(" THIS IS THE SHAPE OF THE IMAGE normalized")
#             # Optionally, you can expand the dimensions to match the input shape expected by the model
#             # For example, if your model expects an input shape of (batch_size, height, width, channels)
#             cropped_image = np.expand_dims(cropped_image, axis=0)  # Adds a batch dimension
#             print(cropped_image.shape)
#             print(" THIS IS THE SHAPE OF THE IMAGE with batch dimension")
#             # Now, 'cropped_image' is ready to be used as input for a CNN model.
#             return cropped_image
#
#     # If there's an issue with the image, return None or handle the error as needed
#     return None


def display_image(cropped_image, class_label="nothing", image_name="nothing"):
    """
    This method is used to display an image
    """
    # Close the image window
    cv2.imshow(class_label, cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
