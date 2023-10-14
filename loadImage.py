import cv2
import os

"""
this method is used to load the image from the file
all it needs is the file name however the folder where the files
are located needs to be in the project directory
"""


def load_image(image_filename):
    image_filename = "phase2_train_v0//final//" + image_filename
    # This gets the current directory where the folder is located
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # This constructs the full path to the image using the current directory
    image_path = os.path.join(current_directory, image_filename)

    # Read the image
    image = cv2.imread(image_path)

    if image is not None:
        return image
    else:
        print(f"Error: The image '{image_filename}' could not be loaded.")
        return None


"""
This method is used to crop the image so only the mosquitoe is returned using the bounding box 
from the csv file as reference, it also resizes the image, grey scales and normalizes it before
returning it
"""


def crop_image(x1, x2, y1, y2, image, target_size=(256, 256)):
    cropped_image = image[y1:y2, x1:x2]

    # Process the cropped image here, e.g., save it or perform further analysis
    # Resize the cropped image to the target size (e.g., 256x256)
    cropped_image = cv2.resize(cropped_image, target_size)
    # Convert the resized image to grayscale
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image = (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min())

    # Close the image window
    # cv2.imshow('Cropped Image', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_image


"""
This method is used to display an image
"""


def display_image(cropped_image, class_label="nothing", image_name="nothing"):
    # Close the image window
    cv2.imshow(class_label, cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#
# # Example usage:
# image_filename = 'phase2_train_v0//final//train_00000.jpeg'  # Just the file name
# loaded_image = load_image(image_filename)
#
# if loaded_image is not None:
#     # You can work with the loaded image here
#     cv2.imshow('Loaded Image', loaded_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # Define the coordinates of the rectangle (x1, y1, x2, y2)
#     x1, y1, x2, y2 = 1301, 1546, 1641, 2096
#
#     # Crop the region from the original image
#     cropped_image = loaded_image[y1:y2, x1:x2]
#     # Display the cropped image (optional)
#     cv2.imshow('Cropped Image', cropped_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
