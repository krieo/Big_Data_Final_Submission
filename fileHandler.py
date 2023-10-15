import csv
import cv2

from ImageData import ImageData


def read_csv_file(file_path):
    """
    This method reads the contents of the file
    and stores it in the class and returns a list
    of the class
    """
    data = []
    with open(file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Create an instance of ImageData for each row in the CSV
            image_data = ImageData(row['img_fName'], row['img_w'], row['img_h'],
                                   row['bbx_xtl'], row['bbx_ytl'], row['bbx_xbr'],
                                   row['bbx_ybr'], row['class_label'])
            data.append(image_data)
    return data