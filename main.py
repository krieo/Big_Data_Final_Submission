import cv2

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
    file_path = "phase2_train_v0.csv"
    image_data_list = fileHandler.read_csv_file(file_path)
    # This reads the data from the file and stores it to a class which is then stored in a list
    for i, image_data in enumerate(image_data_list[:30]):
        print(f"Image {i + 1}: ")
        print(f"File Name: {image_data.img_fName}")
        print(f"Width: {image_data.img_w}")
        print(f"Height: {image_data.img_h}")
        print(
            f"Bounding Box (xtl, ytl, xbr, ybr): ({image_data.bbx_xtl}, {image_data.bbx_ytl}, {image_data.bbx_xbr}, {image_data.bbx_ybr})")
        print(f"Class Label: {image_data.class_label}")
        print()

        # The below methods are used to get an image and save it to the class
        if image_data.class_label == "aegypti":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                #display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_aegypti += 1
                list_aegypti.append(image_data)
        elif image_data.class_label == "albopictus":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                #display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_albopictus += 1
                list_albopictus.append(image_data)
        elif image_data.class_label == "anopheles":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                #display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_anopheles += 1
                list_anopheles.append(image_data)
        elif image_data.class_label == "culex":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                #display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_culex += 1
                list_culex.append(image_data)
        elif image_data.class_label == "culiseta":
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                #display_image(image_data.image, image_data.class_label, image_data.img_fName)
                num_culiseta += 1
                list_culiseta.append(image_data)
        else:
            # This loads the image and crops/resizes/greyscales it to be stored
            loaded_image = load_image(image_data.img_fName)
            if loaded_image is not None:
                new_image = crop_image(image_data.bbx_xtl, image_data.bbx_xbr, image_data.bbx_ytl, image_data.bbx_ybr,
                                       loaded_image)
                image_data.image = new_image
                #display_image(image_data.image, image_data.class_label, image_data.img_fName)
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
#rotated_image = rotate_image(your_input_image, 45)
# Create a list of variables
variables = [num_aegypti, num_albopictus, num_anopheles, num_culex, num_culiseta, num_japonicus_koreicus]

# Find the variable with the highest value
max_value = max(variables)
max_variable = variables.index(max_value)

# Get the name of the variable with the highest value
variable_names = ['num_aegypti', 'num_albopictus', 'num_anopheles', 'num_culex', 'num_culiseta', 'num_japonicus_koreicus']
max_variable_name = variable_names[max_variable]

print(f"The variable with the highest value is: {max_variable_name} = {max_value}")

# This performs class balance for the anopheles
if num_anopheles != 0:
    index = 0
    while num_anopheles < max_value:
        tempItem = list_anopheles[index]
        index += 1
        original_image = tempItem.image
        rotated_image = rotate_image(tempItem.image, 90)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_anopheles.append(tempItem)
        num_anopheles += 1

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
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_aegypti.append(tempItem)
        num_aegypti += 1
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_aegypti.append(tempItem)
        num_aegypti += 1
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_aegypti.append(tempItem)
        num_aegypti += 1

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
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_albopictus.append(tempItem)
        num_albopictus += 1
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_albopictus.append(tempItem)
        num_albopictus += 1
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_albopictus.append(tempItem)
        num_albopictus += 1

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
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_culex.append(tempItem)
        num_culex += 1
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_culex.append(tempItem)
        num_culex += 1
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_culex.append(tempItem)
        num_culex += 1

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
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_culiseta.append(tempItem)
        num_culiseta += 1
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_culiseta.append(tempItem)
        num_culiseta += 1
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_culiseta.append(tempItem)
        num_culiseta += 1

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
        # second round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=False, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test2", "test")
        list_japonicus_koreicus.append(tempItem)
        num_japonicus_koreicus += 1
        # third round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=False)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test3", "test")
        list_japonicus_koreicus.append(tempItem)
        num_japonicus_koreicus += 1
        # fourth round of augmentation
        rotated_image = flip_image(original_image, flip_horizontal=True, flip_vertical=True)
        tempItem.image = rotated_image
        # display_image(tempItem.image, "test4", "test")
        list_japonicus_koreicus.append(tempItem)
        num_japonicus_koreicus += 1

print(str(num_aegypti) + " Num aegypti")
print(str(num_albopictus) + " Num albopictus")
print(str(num_anopheles) + " Num anopheles")
print(str(num_culex) + " Num culex")
print(str(num_culiseta) + " Num culiseta")
print(str(num_japonicus_koreicus) + " Num japonicus")

