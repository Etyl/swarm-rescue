import cv2
import numpy as np
from solutions.mapper.grid import Grid
from spg_overlay.utils.pose import Pose

def display_grid(grid : Grid, robot_pose: Pose, title="grid"):
        """
        Screen display of grid and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """
        img = grid.get_grid().T
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img_color = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        # resize image
        img_color = cv2.resize(img_color, (0, 0), fx=grid.resolution//2, fy=grid.resolution//2, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(title, img_color)
        cv2.waitKey(1)

def save_grid(grid : Grid, title="grid"):
        """
        Save grid and robot pose as an image
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """
        img = grid.get_grid().T
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img_color = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        # resize image
        img_color = cv2.resize(img_color, (0, 0), fx=grid.resolution//2, fy=grid.resolution//2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(title + ".png", img_color)