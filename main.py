import os
import fileHandler
from loadImage import *
import tensorflow as tf

# Constants for batch processing
batch_size = 50  # Change this as needed

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

    # Directory where the subfolders are located
    base_directory = "data"
    boolPerformImageProcessing = False
    # Subfolder names
    subfolders = class_counts.keys()

    for subfolder in subfolders:
        if subfolder == "japonicus/koreicus":
            subfolder = "japonicus_koreicus"
        subfolder_path = os.path.join(base_directory, subfolder)

        # Check if the subfolder exists
        if os.path.exists(subfolder_path):
            print(f"The subfolder '{subfolder}' exists.")

            # Check if the subfolder contains files
            files_in_subfolder = os.listdir(subfolder_path)
            if files_in_subfolder:
                print(f"The subfolder '{subfolder}' contains files.")
            else:
                print(f"The subfolder '{subfolder}' is empty.")
                boolPerformImageProcessing = True
        else:
            print(f"The subfolder '{subfolder}' does not exist.")
            boolPerformImageProcessing = True

if boolPerformImageProcessing == True:

    for i, image_data in enumerate(image_data_list):
        try:
            if image_data.class_label in class_counts:
                processed_image = crop_and_preprocess_image(image_filename + image_data.img_fName,
                                                            image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl,
                                                            image_data.bbx_ybr)

                if processed_image is not None:
                    # Image processing was successful, add it to the lists
                    # image_data.image = processed_image
                    class_counts[image_data.class_label] += 1
                    total_images_processed += 1

                    directory = f"data/{image_data.class_label}"
                    if directory == "data/japonicus/koreicus":
                        directory = "data/japonicus_koreicus"

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Save the image to the directory
                    cv2.imwrite(f"{directory}/{image_data.img_fName}", processed_image)
                    # Free up memory by deleting the processed image
                    del processed_image
                    # Delete the reference to ImageData object
                    del image_data

        except Exception as e:
            print(f"An error occurred while processing image {i}: {e}")

        if (i + 1) % batch_size == 0:
            # Processed one batch, you can perform batch-specific operations here if needed
            print(f"Processed batch {i // batch_size + 1} - Total images processed: {total_images_processed}")

    # Print class counts
    for class_label, count in class_counts.items():
        print(f'Count of {class_label}: {count}')

    print(f'Total images processed: {total_images_processed}')
else:
    print("Files do exist no processing done")

# print(class_lists)
# this builds the image data set on the fly it also does preprocessing
data = tf.keras.utils.image_dataset_from_directory('data')
# print(data.count())
# this allows us to access the data from the pipeline
# data_iterator = data.as_numpy_iterator()
# batch = data_iterator.next()
# batch = data_iterator.next() this command can be run multiple times to get the next batch
# this prints a 2 which is the images and the labels images are in key 0 batch[0] and labels are in key 1 batch[1]
# print(len(batch))
# this is the images as represented as a numpy array
# print(batch[0].shape)
# this is the labels for the images as normalized
# print(batch[1])
# this normalises the values to a range of 0 and 1
# scaled_batch = batch[0] / 255
# print(scaled_batch.min())
# print(scaled_batch.max())
# this is a more efficient way to normalize the data as it does it when loaded in pipeline
data = data.map(lambda x, y: (x/255, y))
# print(data.as_numpy_iterator().next()[0].min())

# split the data in training and testing sets
# the train and validation sets will be used during training
print(len(data))
train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.1) + 2
print("Sizes:")
print(train_size)
print(val_size)
print(test_size)

# This allows for data to be split into chucks on the size calculated earlier
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
print(len(train))
print(len(val))
print(len(test))
