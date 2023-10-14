import cv2
import os

def load_image(image_filename):
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


# Example usage:
image_filename = 'phase2_train_v0//final//train_00000.jpeg'  # Just the file name
loaded_image = load_image(image_filename)

if loaded_image is not None:
    # You can work with the loaded image here
    cv2.imshow('Loaded Image', loaded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
