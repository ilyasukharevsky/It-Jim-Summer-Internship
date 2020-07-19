"""
Perhaps, it would be rewarding to work with separate channels,
but for now I cannot offer a better solution, unfortunately.
"""

import cv2
import numpy as np


class ColorRange:
    def __init__(self, name, low, high):
        self.name = name
        self.low = low
        self.high = high


class ExtractGeometry:
    def __init__(self, input_video_name, output_video_name, video_writer_type, colors_to_extract, video_delay):
        self.cap = cv2.VideoCapture(input_video_name)
        self.colors_to_extract = colors_to_extract
        self.video_delay = video_delay
        ret, frm = self.cap.read()
        self.writer = cv2.VideoWriter(output_video_name, video_writer_type, 30, (frm.shape[1], frm.shape[0]), 0)

    def run(self):
        ret, frm = self.cap.read()
        key = None

        while ret and key != ord(' '):
            hsv_frm = cv2.cvtColor(frm, cv2.COLOR_RGB2HSV)
            hsv_frm = cv2.GaussianBlur(hsv_frm, (5, 5), 3)

            submasks = self.collect_submasks(hsv_frm)
            full_mask = self.combine_masks(submasks)

            self.writer.write(full_mask)
            cv2.imshow('Video frame', full_mask)

            ret, frm = self.cap.read()
            key = cv2.waitKey(self.video_delay)


        self.clean_up()

    def collect_submasks(self, hsv_frm):
        submasks = []

        for color in self.colors_to_extract:
            submask = cv2.inRange(hsv_frm, color.low, color.high)
            submasks.append(submask)
        return submasks

    def combine_masks(self, submasks):
        full_mask = submasks[0]
        for submask in submasks[1:]:
            full_mask |= submask
        return full_mask

    def filter_noise(self, submask):
        pass

    def clean_up(self):
        self.cap.release()
        self.writer.release()


if __name__ == '__main__':
    colors = [ColorRange("green", np.array([20, 40, 100]), np.array([80, 100, 255])),
              ColorRange("blue", np.array([94, 80, 2]), np.array([126, 255, 255])),
              ColorRange("black", np.array([0, 0, 0]), np.array([180, 255, 50])),
              ColorRange("red", np.array([170, 120, 70]), np.array([180, 255, 255]))]

    """[ColorRange("green", np.array([25, 52, 72]), np.array([102, 255, 255])),
     ColorRange("blue", np.array([94, 80, 2]), np.array([126, 255, 255])),
     ColorRange("black", np.array([0, 0, 0]), np.array([180, 255, 50])),
     ColorRange("red", np.array([170, 120, 70]), np.array([180, 255, 255]))]
    mask_gy = cv2.inRange(frm_hsv, np.array([20, 40, 100]), np.array([80, 100, 255]))

    # mask for pink
    mask_p = cv2.inRange(frm_hsv, np.array([140, 50, 0]), np.array([179, 150, 255]))

    # mask for black
    mask_b = cv2.inRange(frm_hsv, np.array([0, 0, 0]), np.array([179, 255, 40]))"""
    extractor = ExtractGeometry("input_video.avi", "output_video.avi",
                                video_writer_type=cv2.VideoWriter_fourcc(*"XVID"),
                                colors_to_extract=colors, video_delay=10)
    extractor.run()


