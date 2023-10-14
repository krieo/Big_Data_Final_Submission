""""
This class holds all of the information
for each of the images
"""


class ImageData:
    def __init__(self, img_fName, img_w, img_h, bbx_xtl, bbx_ytl, bbx_xbr, bbx_ybr, class_label, image=None):
        self.img_fName = img_fName
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.bbx_xtl = int(bbx_xtl)
        self.bbx_ytl = int(bbx_ytl)
        self.bbx_xbr = int(bbx_xbr)
        self.bbx_ybr = int(bbx_ybr)
        self.class_label = class_label
        self.image = None
