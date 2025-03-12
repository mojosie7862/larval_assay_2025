import numpy as np
import cv2

class Larvae():
    def __init__(self, frame, img_property_dict):
        self.frame = frame
        self.centroid = img_property_dict.centroid
        self.coords = img_property_dict.coords
        self.row_i = int(self.centroid[0] * frame.n_well_rows / frame.n_pixels)
        self.col_i = int(self.centroid[1] * frame.n_well_cols / frame.n_pixels)
        self.larvae_i = (self.row_i * frame.n_well_cols) + self.col_i
        self.y_pixel, self.x_pixel = self.centroid
        self.x_pixel_cropped = self.x_pixel / (self.col_i + 1)
        self.y_pixel_cropped = self.y_pixel / (self.row_i + 1)
        self.x_um = self.x_pixel * self.frame.pixel_um
        self.y_um = self.y_pixel * self.frame.pixel_um
        self.x_um_cropped = self.x_um / (self.col_i)
        self.y_um_cropped = self.y_um / (self.row_i)

        self.larvae_img = self.crop_larvae_img()

    def crop_larvae_img(self):
        x0 = int(self.frame.n_pixels / self.frame.n_well_rows) * self.col_i
        x1 = int(self.frame.n_pixels / self.frame.n_well_rows) * (self.col_i + 1)
        y0 = int(self.frame.n_pixels / self.frame.n_well_rows) * self.row_i
        y1 = int(self.frame.n_pixels / self.frame.n_well_rows) * (self.row_i + 1)

        larvae_img = self.frame.thresholded_img[y0:y1, x0:x1]

        return larvae_img

    def amplitude_metrics(self):
        print(self.x_pixel_cropped, self.y_pixel_cropped)
        # rr, cc = circle_perimeter(int(self.y_pixel_cropped), int(self.x_pixel_cropped), 1)
        # self.larvae_img[rr, cc] = 255

        # font = cv2.FONT_HERSHEY_PLAIN
        # cv2.putText(frame.img, str(k),
        #             (int(x), int(y)), font,
        #             1.5,
        #             (255, 255, 255),
        #             2, cv2.LINE_8)


        headInd = np.argmax(self.frame.img[self.coords[:, 0], self.coords[:, 1]])
        # y_0 = coords[headInd, 0]
        # x_0 = coords[headInd, 1]
        # print(self.coords)
        print(self.larvae_i, headInd)
        cv2.imwrite('larvae_images/cropped_larvae_image_'+str(self.larvae_i)+'.png', self.larvae_img)

        # rr, cc = circle_perimeter(int(y), int(x), 1)
        # frame.img[rr, cc] = 255

        # font = cv2.FONT_HERSHEY_PLAIN
        # cv2.putText(frame.img, str(k),
        #             (int(x), int(y)), font,
        #             1.5,
        #             (255, 255, 255),
        #             2, cv2.LINE_8)

        # cv2.imwrite('amplitude_dev_' + self.frame_i + '.png', self.frame_i)