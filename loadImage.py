import cv2
import os


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


def crop_image(x1, x2, y1, y2, image):
    cropped_image = image[y1:y2, x1:x2]

    # Process the cropped image here, e.g., save it or perform further analysis

    # Close the image window
    cv2.imshow('Cropped Image', cropped_image)
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
