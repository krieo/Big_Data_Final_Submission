import cv2
import os
from PIL import Image
import numpy as np


def load_image(image_filename):
    """
    this method is used to load the image from the file
    all it needs is the file name however the folder where the files
    are located needs to be in the project directory
    """
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


def crop_image(x1, x2, y1, y2, image, target_size=(256, 256)):
    """
    This method is used to crop the image so only the mosquito is returned using the bounding box
    from the CSV file as reference, it also resizes the image, grayscales and normalizes it before
    returning it.
    """
    try:
        # Attempt to crop the image using the first set of coordinates (x1, x2, y1, y2)
        cropped_image = image[y1:y2, x1:x2]

        # Process the cropped image here, e.g., save it or perform further analysis
        # Resize the cropped image to the target size (e.g., 256x256)
        cropped_image = cv2.resize(cropped_image, target_size)
        # Convert the resized image to grayscale
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min())

        return cropped_image
    except Exception as e:
        try:
            # Attempt to crop the image using the first set of coordinates (x1, x2, y1, y2)
            cropped_image = image[y2:y1, x2:x1]

            # Process the cropped image here, e.g., save it or perform further analysis
            # Resize the cropped image to the target size (e.g., 256x256)
            cropped_image = cv2.resize(cropped_image, target_size)
            # Convert the resized image to grayscale
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            cropped_image = (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min())

            return cropped_image
        except Exception as e2:
            return image


def flip_image(image, flip_horizontal=False, flip_vertical=False):
    """
    This flips an image horizontally, vertically, or both.

    Args:
        image: The input image to be flipped.
        flip_horizontal: Whether to perform horizontal flipping.
        flip_vertical: Whether to perform vertical flipping.

    Returns:
        flipped_image: The flipped image.
    """
    flipped_image = image.copy()

    if flip_horizontal:
        flipped_image = cv2.flip(flipped_image, 1)  # 1 for horizontal flip

    if flip_vertical:
        flipped_image = cv2.flip(flipped_image, 0)  # 0 for vertical flip

    return flipped_image


def rotate_image(image, angle_degrees):
    """
    This method rotates the image by the number of degrees

    Args:
        image: The input image to be rotated.
        angle_degrees: The angle in degrees by which to rotate the image.

    Returns:
        rotated_image: The rotated image.
    """
    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def adjust_brightness_contrast(image, alpha, beta):
    """
    This adjusts the brightness and contrast of an image.

    Args:
        image: The input image.
        alpha: Controls the contrast (1.0 for no change).
        beta: Controls the brightness (0 for no change).

    Returns:
        adjusted_image: The image with adjusted brightness and contrast.
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def inject_noise(image, noise_type='gaussian', mean=0, std=25):
    """
    This method is used to inject noise into an image.

    Args:
        image: The input image.
        noise_type: Type of noise to inject gaussian or speckle by default i am only using gaussian
        mean: Mean value for the noise that is used
        std: Standard deviation for the noise used in gaussian.

    Returns:
        noisy_image: The image with injected noise.
    """
    noisy_image = image.copy()

    if noise_type == 'gaussian':
        h, w, = noisy_image.shape
        c = 1
        noise = np.random.normal(mean, std, (h, w, c))
        noisy_image = cv2.add(image, noise)
    elif noise_type == 'speckle':
        h, w, = noisy_image.shape
        c = 1
        noise = np.random.normal(mean, std, (h, w, c))
        noisy_image += noisy_image * noise

    return noisy_image


def display_image(cropped_image, class_label="nothing", image_name="nothing"):
    """
    This method is used to display an image
    """
    # Close the image window
    cv2.imshow(class_label, cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess_image(image_data):
    try:
        # Assuming image_data is a PIL Image or an image in a suitable format
        # Resize the image to the desired size (e.g., 256x256)
        image = image_data.resize((256, 256))
        # Convert the image to a NumPy array
        image_array = np.array(image)
        # Normalize pixel values (if needed)
        image_array = image_array / 255.0  # Normalize to [0, 1] range
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
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
