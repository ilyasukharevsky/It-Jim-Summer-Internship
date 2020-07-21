import cv2
import numpy as np
from math import sqrt


class ColorRange:
    def __init__(self, name, low, high):
        self.name = name
        self.low = low
        self.high = high


def first_match(iterable, predicate):
    try:
        return next(elem for elem in iterable if predicate(elem))
    except StopIteration:
        return None


class ExtractGeometry:

    def __init__(self, input_video_name, output_video_name, video_writer_type, colors_to_extract, video_delay):
        self.cap = cv2.VideoCapture(input_video_name)
        self.colors_to_extract = colors_to_extract
        self.video_delay = video_delay
        ret, frm = self.cap.read()
        self.writer = cv2.VideoWriter(output_video_name, video_writer_type, 30, (frm.shape[1], frm.shape[0]))

    def show_and_write_contours(self):
        """
        A duplicated contour is detected as a contour having a center close to the considered one.
        Duplicated contours are removed from the array of contours.
        """
        ret, img = self.cap.read()
        key = None

        while ret and key != ord(' '):
            mask = self.get_black_and_white_mask(img)

            conts, _ = self.find_contours(mask)
            conts = self.remove_contours_with_same_center(conts)
            self.draw_conts(img, conts)
            self.tag_conts(img, conts)

            self.write_and_show(img, "Contour geometry")

            ret, img = self.cap.read()
            key = cv2.waitKey(self.video_delay)

        self.clean_up()

    def show_and_write_contours_2(self):
        """
        External duplicated contours are removed by using masks. A more successful attempt.
        """
        ret, img = self.cap.read()
        key = None

        while ret and key != ord(' '):
            pure_img = img.copy()
            mask = self.get_black_and_white_mask(img)

            mask_copy = mask.copy()
            conts, _ = self.find_contours(mask_copy)

            self.draw_conts(img, conts)

            img_masked_outside_figures = cv2.bitwise_and(img, img, mask=mask)
            inv_mask = 255 - mask
            img_masked_inside_figures = cv2.bitwise_and(pure_img, pure_img, mask=inv_mask)

            final_img = img_masked_outside_figures + img_masked_inside_figures
            self.tag_conts(final_img, conts)

            self.write_and_show(final_img, "Contour geometry")

            ret, img = self.cap.read()
            key = cv2.waitKey(self.video_delay)

        self.clean_up()

    def show_and_write_black_and_white_mask(self):
        """
        Task_1 IT-JIM
        """
        ret, frm = self.cap.read()
        key = None

        while ret and key != ord(' '):
            hsv_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)

            submasks = self.collect_submasks(hsv_frm)
            full_mask = self.combine_masks(submasks)

            self.write_and_show(full_mask, "Black and white geometry")

            ret, frm = self.cap.read()
            key = cv2.waitKey(self.video_delay)

        self.clean_up()

    def remove_contours_with_same_center(self, cnts):
        i, j = 0, 0
        while i < len(cnts):
            num = -1
            while j < len(cnts):
                if i != j and self.dist_between_centers(cnts[i], cnts[j]) < 300:
                    num = j
                    break
                j += 1
            i += 1
            if num >= 0:
                del(cnts[num])
        return cnts

    def remove_contours_with_children(self, cnts, hiers):
        i, j = 0, 0
        while i < len(cnts):
            if not hiers[0][i][2] == -1:
                del(cnts[i])
                j += 1
            else:
                i += 1
                j += 1
        return cnts

    def get_black_and_white_mask(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        submasks = self.collect_submasks(hsv_img)
        full_mask = self.combine_masks(submasks)
        return full_mask

    def collect_submasks(self, hsv_frm):
        submasks = []
        for color in self.colors_to_extract:
            submask = cv2.inRange(hsv_frm, color.low, color.high)
            submasks.append(submask)
        return submasks

    def combine_masks(self, submasks):
        full_mask = submasks[0]
        for submask in submasks[1:]:
            full_mask += submask
        return full_mask

    def find_contours(self, img):
        img_thresh = cv2.adaptiveThreshold(img[:, :], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        img_thresh = cv2.dilate(img_thresh, np.ones((3, 3)))
        img_thresh = cv2.erode(img_thresh, np.ones((11, 11)))
        conts, hier = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return conts, hier

    def draw_conts(self, img, conts):
        figure_to_color = {"Triangle": np.array([0, 0, 255]),
                           "Rectangle": np.array([255, 0, 0]),
                           "Circle": np.array([0, 255, 0])}
        for cnt in conts:
            if self.is_right_size_and_not_near_borders(img, cnt):
                figure = self.get_figure_name(cnt)
                clr = figure_to_color[figure]
                cv2.drawContours(img, [cnt], 0, clr.tolist(), 4)

    def tag_conts(self, img, conts):
        for cnt in conts:
            if self.is_right_size_and_not_near_borders(img, cnt):
                self.put_figure_name(img, cnt)

    def is_right_size_and_not_near_borders(self, img, cnt):
        return 700 < cv2.contourArea(cnt) < 17000 and not self.contour_near_image_borders(img, cnt, 25)

    def contour_near_image_borders(self, img, cnt, dist):
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min, x_max, y_max = 0, 0, img.shape[1] - 1, img.shape[0] - 1
        if x_max - dist <= x + w or x <= x_min + dist or y_max - dist <= y + h or y <= y_min + dist:
            return True
        return False

    def get_figure_name(self, cnt):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04*peri, True)

        if len(approx) == 3:
            figure_name = "Triangle"
        elif len(approx) == 4:
            figure_name = "Rectangle"
        else:
            figure_name = "Circle"

        return figure_name

    def put_figure_name(self, img, cnt):
        figure = self.get_figure_name(cnt)
        cv2.putText(img, figure, self.contour_center(cnt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    def write_and_show(self, img, title):
        cv2.imshow(title, img)
        self.writer.write(img)

    def contour_center(self, cnt):
        m = cv2.moments(cnt)
        if m["m00"] != 0:
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])
            return c_x, c_y
        return None, None

    def dist_between_centers(self, cnt_1, cnt_2):
        _, _, w, h = cv2.boundingRect(cnt_1)
        x1, y1 = w/2, h/2
        _, _, w, h = cv2.boundingRect(cnt_2)
        x2, y2 = w/2, h/2

        return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

    def clean_up(self):
        self.cap.release()
        self.writer.release()


if __name__ == '__main__':
    colors = [ColorRange("green_yellow", np.array([20, 40, 100]), np.array([80, 100, 255])),
              ColorRange("pink", np.array([140, 50, 0]), np.array([179, 150, 255])),
              ColorRange("black", np.array([0, 0, 0]), np.array([179, 255, 40]))]

    extractor = ExtractGeometry("input_video.avi", "output_video.avi",
                                video_writer_type=cv2.VideoWriter_fourcc(*"XVID"),
                                colors_to_extract=colors, video_delay=20)
    extractor.show_and_write_contours_2()
