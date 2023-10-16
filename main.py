import os
import fileHandler
from loadImage import *

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

    for i, image_data in enumerate(image_data_list):
        if image_data.class_label in class_counts:
            processed_image = crop_and_preprocess_image(image_filename + image_data.img_fName,
                                                        image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl,
                                                        image_data.bbx_ybr)

            if processed_image is not None:
                # Image processing was successful, add it to the lists
                image_data.image = processed_image
                class_counts[image_data.class_label] += 1
                #class_lists[image_data.class_label].append(image_data)
                total_images_processed += 1

                directory = f"data/{image_data.class_label}"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Save the image to the directory
                cv2.imwrite(f"{directory}/{image_data.img_fName}", processed_image)

        if (i + 1) % batch_size == 0:
            # Processed one batch, you can perform batch-specific operations here if needed
            print(f"Processed batch {i // batch_size + 1} - Total images processed: {total_images_processed}")

    # Print class counts
    for class_label, count in class_counts.items():
        print(f'Count of {class_label}: {count}')

    print(f'Total images processed: {total_images_processed}')

#print(class_lists)

