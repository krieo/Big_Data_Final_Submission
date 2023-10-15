import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import fileHandler
from loadImage import *

# These are the variables to count how many instances there are
num_aegypti = 0
num_albopictus = 0
num_anopheles = 0
num_culex = 0
num_culiseta = 0
num_japonicus_koreicus = 0

# These are the seperate list that will hold the data for each of its respective classes
list_aegypti = []
list_albopictus = []
list_anopheles = []
list_culex = []
list_culiseta = []
list_japonicus_koreicus = []

if __name__ == '__main__':
    # output_folder = "culex"
    # this creates the output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)
    file_path = "phase2_train_v0.csv"
    image_data_list = fileHandler.read_csv_file(file_path)
    # This reads the data from the file and stores it to a class which is then stored in a list
    for i, image_data in enumerate(image_data_list[:200]):
        # print(f"Image {i + 1}: ")
        # print(f"File Name: {image_data.img_fName}")
        # print(f"Width: {image_data.img_w}")
        # print(f"Height: {image_data.img_h}")
        # print(
        #   f"Bounding Box (xtl, ytl, xbr, ybr): ({image_data.bbx_xtl}, {image_data.bbx_ytl}, {image_data.bbx_xbr}, {image_data.bbx_ybr})")
        # print(f"Class Label: {image_data.class_label}")
        # print()

        # The below methods are used to get an image and save it to the class
        if image_data.class_label == "aegypti":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                # display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_aegypti += 1
                list_aegypti.append(image_data)
        elif image_data.class_label == "albopictus":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                # display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_albopictus += 1
                list_albopictus.append(image_data)
        elif image_data.class_label == "anopheles":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                # display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_anopheles += 1
                list_anopheles.append(image_data)
        elif image_data.class_label == "culex":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                # display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_culex += 1
                list_culex.append(image_data)
                # Define a unique filename for the rotated image (e.g., using an identifier)
                # rotated_filename = f"image" + str(num_culex) + ".jpeg"
                # Define the full path to save the rotated image
                # rotated_image_path = os.path.join(output_folder, rotated_filename)
                # Save the rotated image, overwriting if it already exists
                # cv2.imwrite(rotated_image_path, new_image)
        elif image_data.class_label == "culiseta":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                # display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_culiseta += 1
                list_culiseta.append(image_data)
        else:
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                # display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_japonicus_koreicus += 1
                list_japonicus_koreicus.append(image_data)

# This is just used to print some information to the screen
print(num_aegypti)
print(num_albopictus)
print(num_anopheles)
print(num_culex)
print(num_culiseta)
print(num_japonicus_koreicus)
sum1 = num_aegypti + num_albopictus + num_anopheles + num_culex + num_culiseta + num_japonicus_koreicus
print(sum1)

# Count the number of elements in each list
count_aegypti = len(list_aegypti)
count_albopictus = len(list_albopictus)
count_anopheles = len(list_anopheles)
count_culex = len(list_culex)
count_culiseta = len(list_culiseta)
count_japonicus_koreicus = len(list_japonicus_koreicus)

# Print the counts
print(f'Count of aegypti: {count_aegypti}')
print(f'Count of albopictus: {count_albopictus}')
print(f'Count of anopheles: {count_anopheles}')
print(f'Count of culex: {count_culex}')
print(f'Count of culiseta: {count_culiseta}')
print(f'Count of japonicus_koreicus: {count_japonicus_koreicus}')

# This part adds more data augmentation so there is no class imbalance
# rotated_image = rotate_image(your_input_image, 45)
# Create a list of variables
variables = [num_aegypti, num_albopictus, num_anopheles, num_culex, num_culiseta, num_japonicus_koreicus]

# Find the variable with the highest value
max_value = max(variables)
max_variable = variables.index(max_value)

# Get the name of the variable with the highest value
variable_names = ['num_aegypti', 'num_albopictus', 'num_anopheles', 'num_culex', 'num_culiseta',
                  'num_japonicus_koreicus']
max_variable_name = variable_names[max_variable]

print(f"The variable with the highest value is: {max_variable_name} = {max_value}")

# This performs class balance for the anopheles
if num_anopheles != 0:
    index = 0
    # output_folder = "anopheles"
    # this creates the output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)
    while num_anopheles < max_value:
        tempItem = list_anopheles[index]
        index += 1
        original_image = tempItem.image
        # Define a unique filename for the rotated image (e.g., using an identifier)
        # rotated_filename = f"image" + str(index) + str(num_anopheles) + ".jpeg"
        # Define the full path to save the rotated image
        # rotated_image_path = os.path.join(output_folder, rotated_filename)
        # Save the rotated image, overwriting if it already exists
        # cv2.imwrite(rotated_image_path, original_image)

        rotated_image = rotate_image(tempItem.image, 90)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1
        print("anopheles " + str(num_anopheles))
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1
        print("anopheles " + str(num_anopheles))
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1
        print("anopheles " + str(num_anopheles))
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1
        print("anopheles " + str(num_anopheles))

# This performs class balance for the aegypti
if num_aegypti != 0:
    index = 0
    while num_aegypti < max_value:
        tempItem = list_aegypti[index]
        index += 1
        original_image = tempItem.image
        rotated_image = rotate_image(tempItem.image, 90)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test", "test")
        list_aegypti.append(tempItem)
        num_aegypti += 1
        print("aegypti " + str(num_aegypti))
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_aegypti.append(tempItem)
        num_aegypti += 1
        print("aegypti " + str(num_aegypti))
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_aegypti.append(tempItem)
        num_aegypti += 1
        print("aegypti " + str(num_aegypti))
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_aegypti.append(tempItem)
        num_aegypti += 1
        print("aegypti " + str(num_aegypti))

# This performs class balance for the albopictus
if num_albopictus != 0:
    index = 0
    while num_albopictus < max_value:
        tempItem = list_albopictus[index]
        index += 1
        original_image = tempItem.image
        rotated_image = rotate_image(tempItem.image, 90)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test", "test")
        list_albopictus.append(tempItem)
        num_albopictus += 1
        print("albopictus " + str(num_albopictus))
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_albopictus.append(tempItem)
        num_albopictus += 1
        print("albopictus " + str(num_albopictus))
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_albopictus.append(tempItem)
        num_albopictus += 1
        print("albopictus " + str(num_albopictus))
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_albopictus.append(tempItem)
        num_albopictus += 1
        print("albopictus " + str(num_albopictus))

# This performs class balance for the culex
if num_culex != 0:
    index = 0
    while num_culex < max_value:
        tempItem = list_culex[index]
        index += 1
        original_image = tempItem.image
        rotated_image = rotate_image(tempItem.image, 90)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test", "test")
        list_culex.append(tempItem)
        num_culex += 1
        print("culex " + str(num_culex))
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_culex.append(tempItem)
        num_culex += 1
        print("culex " + str(num_culex))
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_culex.append(tempItem)
        num_culex += 1
        print("culex " + str(num_culex))
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_culex.append(tempItem)
        num_culex += 1
        print("culex " + str(num_culex))

# This performs class balance for the culiseta
if num_culiseta != 0:
    index = 0
    while num_culiseta < max_value:
        tempItem = list_culiseta[index]
        index += 1
        original_image = tempItem.image
        rotated_image = rotate_image(tempItem.image, 90)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test", "test")
        list_culiseta.append(tempItem)
        num_culiseta += 1
        print("culiseta " + str(num_culiseta))
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_culiseta.append(tempItem)
        num_culiseta += 1
        print("culiseta " + str(num_culiseta))
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_culiseta.append(tempItem)
        num_culiseta += 1
        print("culiseta " + str(num_culiseta))
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_culiseta.append(tempItem)
        num_culiseta += 1
        print("culiseta " + str(num_culiseta))

# This performs class balance for the japonicus_koreicus
if num_japonicus_koreicus != 0:
    index = 0
    while num_japonicus_koreicus < max_value:
        tempItem = list_japonicus_koreicus[index]
        index += 1
        original_image = tempItem.image
        rotated_image = rotate_image(tempItem.image, 90)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test", "test")
        list_japonicus_koreicus.append(tempItem)
        num_japonicus_koreicus += 1
        print("japonicus_koreicus " + str(num_japonicus_koreicus))
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_japonicus_koreicus.append(tempItem)
        num_japonicus_koreicus += 1
        print("japonicus_koreicus " + str(num_japonicus_koreicus))
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_japonicus_koreicus.append(tempItem)
        num_japonicus_koreicus += 1
        print("japonicus_koreicus " + str(num_japonicus_koreicus))
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_japonicus_koreicus.append(tempItem)
        num_japonicus_koreicus += 1
        print("japonicus_koreicus " + str(num_japonicus_koreicus))

print(str(num_aegypti) + " Num aegypti")
print(str(num_albopictus) + " Num albopictus")
print(str(num_anopheles) + " Num anopheles")
print(str(num_culex) + " Num culex")
print(str(num_culiseta) + " Num culiseta")
print(str(num_japonicus_koreicus) + " Num japonicus")

# this performs the split
test_size = 0.2  # this ensures a 80 to 20 split
random_state = 42  # this plants a random seed so its always the same split
# Split the list for each class
train_aegypti, test_aegypti = train_test_split(list_aegypti, test_size=test_size, random_state=random_state)
train_albopictus, test_albopictus = train_test_split(list_albopictus, test_size=test_size, random_state=random_state)
train_anopheles, test_anopheles = train_test_split(list_anopheles, test_size=test_size, random_state=random_state)
train_culex, test_culex = train_test_split(list_culex, test_size=test_size, random_state=random_state)
train_culiseta, test_culiseta = train_test_split(list_culiseta, test_size=test_size, random_state=random_state)
train_japonicus_koreicus, test_japonicus_koreicus = train_test_split(list_japonicus_koreicus, test_size=test_size, random_state=random_state)

train_data = [(item.image, item.class_label) for item in train_aegypti] + \
             [(item.image, item.class_label) for item in train_albopictus] + \
             [(item.image, item.class_label) for item in train_anopheles] + \
             [(item.image, item.class_label) for item in train_culex] + \
             [(item.image, item.class_label) for item in train_culiseta] + \
             [(item.image, item.class_label) for item in train_japonicus_koreicus]

test_data = [(item.image, item.class_label) for item in test_aegypti] + \
            [(item.image, item.class_label) for item in test_albopictus] + \
            [(item.image, item.class_label) for item in test_anopheles] + \
            [(item.image, item.class_label) for item in test_culex] + \
            [(item.image, item.class_label) for item in test_culiseta] + \
            [(item.image, item.class_label) for item in test_japonicus_koreicus]

# Iterate through training data and print class labels
print("Training Data Class Labels:")
for image, label in train_data:
    print(label)

# Iterate through testing data and print class labels
print("\nTesting Data Class Labels:")
for image, label in test_data:
    print(label)

# Preprocess the training data
preprocessed_train_data = [(preprocess_image(image), label) for image, label in train_data]

# Preprocess the testing data
preprocessed_test_data = [(preprocess_image(image), label) for image, label in test_data]

# Create a LabelBinarizer
label_binarizer = LabelBinarizer()

# Fit the LabelBinarizer on the training labels
label_binarizer.fit([label for _, label in preprocessed_train_data])

# Transform the training and testing labels to one-hot encoded vectors
y_train = label_binarizer.transform([label for _, label in preprocessed_train_data])
y_test = label_binarizer.transform([label for _, label in preprocessed_test_data])

print("One-Hot Encoded Training Labels:")
print(y_train)

print("\nOne-Hot Encoded Testing Labels:")
print(y_test)
