import cv2
import numpy as np
import pygame


class ScreenRecorder:
    """
    This class is used to record a PyGame surface and save it to a video file.
    """

    def __init__(self, width, height, fps, out_file="output.avi"):
        """
        Initialize the recorder with parameters of the surface.
        :param width: Width of the surface to capture
        :param height: Height of the surface to capture
        :param fps: Frames per second
        :param out_file: Output file to save the recording
        """
        print(
            f"Initializing ScreenRecorder with parameters width:{width} height:{height} fps:{fps}."
        )
        print(f"Output of the screen recording saved to {out_file}.")

        # define the codec and create a video writer object
        four_cc = cv2.VideoWriter_fourcc(*"XVID")
        self.width = width
        self.height = height
        self.video = cv2.VideoWriter(out_file, four_cc, float(fps), (width, height))

    def capture_frame(self, surf, text=None, overlay=None):
        """
         Call this method every frame, pass in the pygame surface to capture.
        :param surf: pygame surface to capture
        :return: None
        """
        """

            Note: surface must have the dimensions specified in the constructor.
        """
        # transform the pixels to the format used by open-cv
        pixels = cv2.rotate(pygame.surfarray.pixels3d(surf), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.resize(pixels, (self.width, self.height))
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        if text is not None:
            for i, line in enumerate(text.split("\n")):
                y = 50 + i * 50
                pixels = cv2.putText(
                    pixels,
                    line,
                    org=(10, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        if overlay is not None:
            pixels = np.where(overlay == [255, 255, 255], pixels, overlay)

        # write the frame
        self.video.write(pixels)

    def end_recording(self):
        """
        Call this method to stop recording.
        :return: None
        """
        # stop recording
        self.video.release()
