import os
import fileHandler
from loadImage import *
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import random
# Constants for batch processing
batch_size = 100  # Change this as needed

# These are the variables to count how many instances there are
class_counts = {
    "aegypti": 0,
    "albopictus": 0,
    "anopheles": 0,
    "culex": 0,
    "culiseta": 0,
    "japonicus/koreicus": 0
}

# These are the separate lists that will hold the data for each of its respective classes
class_lists = {
    "aegypti": [],
    "albopictus": [],
    "anopheles": [],
    "culex": [],
    "culiseta": [],
    "japonicus/koreicus": []
}

image_filename = "phase2_train_v0//final//"
if __name__ == '__main__':
    file_path = "phase2_train_v0.csv"
    image_data_list = fileHandler.read_csv_file(file_path)
    total_images_processed = 0

    # Create a new list to store the items you want to keep
    new_image_data_list = []

    for i, image_data in enumerate(image_data_list[:100]):
        if image_data.class_label in class_counts:
            processed_image = crop_and_preprocess_image(image_filename + image_data.img_fName,
                                                        image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl,
                                                        image_data.bbx_ybr)

            if processed_image is not None:
                # Image processing was successful, add it to the lists
                image_data.image = processed_image
                class_counts[image_data.class_label] += 1
                class_lists[image_data.class_label].append(image_data)
                total_images_processed += 1
                new_image_data_list.append(image_data)  # Add to the new list

        if (i + 1) % batch_size == 0:
            # Processed one batch, you can perform batch-specific operations here if needed
            print(f"Processed batch {i // batch_size + 1} - Total images processed: {total_images_processed}")

    # Replace image_data_list with the new list
    image_data_list = new_image_data_list

    # Print class counts
    for class_label, count in class_counts.items():
        print(f'Count of {class_label}: {count}')

    print(f'Total images processed: {total_images_processed}')
label_encoder = LabelEncoder()
class_labels = [data.class_label for data in image_data_list]
encoded_labels = label_encoder.fit_transform(class_labels)
one_hot_labels = to_categorical(encoded_labels)


X = [data.image for data in image_data_list]
y = one_hot_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)

# Print original class labels
print("Original Class Labels:")
print(class_labels)

# Print encoded labels
print("Encoded Labels:")
print(encoded_labels)

# Print one-hot encoded labels
print("One-Hot Encoded Labels:")
print(one_hot_labels)

# Combine images and labels
image_label_pairs = list(zip(X_train, y_train))

# Now, image_label_pairs is a list of tuples, where each tuple contains an image (X_train) and its label (y_train).

# Shuffle the data to ensure randomization during training (optional)

random.shuffle(image_label_pairs)

# Unzip the shuffled data back into separate lists for training
X_train_shuffled, y_train_shuffled = zip(*image_label_pairs)

# Convert to NumPy arrays
X_train_shuffled = np.array(X_train_shuffled)
y_train_shuffled = np.array(y_train_shuffled)

# Display the first 5 images with their labels
for i in range(3):
    image = (X_train_shuffled[i][0] * 255).astype(np.uint8)  # Convert values back to 0-255 range
    label = y_train_shuffled[i]

    # Decode the one-hot label back to its original class label
    original_label = label_encoder.inverse_transform([label.argmax()])[0]

    # Debugging information to check image dimensions and data type
    print(f"Image {i} - Shape: {image.shape}, Data Type: {image.dtype}")

    # Display the image with its original label
    cv2.imshow(f'Label: {original_label}', image)

# Wait until a key is pressed and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()


