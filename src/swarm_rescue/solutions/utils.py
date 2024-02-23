import numpy as np
import arcade

map_id_to_color = {
    0: arcade.color.BLACK,
    1: arcade.color.WHITE,
    2: arcade.color.GREEN,
    3: arcade.color.RED,
    4: arcade.color.BLUE,
    5: arcade.color.YELLOW,
    6: arcade.color.PURPLE,
    7: arcade.color.ORANGE,
    8: arcade.color.PINK,
    9: arcade.color.BROWN,
    10: arcade.color.GRAY,
    11: arcade.color.LIGHT_GRAY,
    12: arcade.color.DARK_GREEN,
    13: arcade.color.DARK_RED,
}

def normalize_angle(angle, zero_2_2pi=False):
    """
      Angle modulo operation
      Default angle modulo range is [-pi, pi)

      Parameters
      ----------
      angle : float or array_like
          A angle or an array of angles. This array is flattened for
          the calculation. When an angle is provided, a float angle is returned.
      zero_2_2pi : bool, optional
          Change angle modulo range to [0, 2pi)
          Default is False.

      Returns
      -------
      ret : float or ndarray
          an angle or an array of modulated angle.

      Examples
      --------
      >>> normalize_angle(-4.0)
      2.28318531

      >>> normalize_angle([-4.0])
      np.array(2.28318531)

      """
    if isinstance(angle, float) or isinstance(angle, int):
        is_float = True
    else:
        is_float = False

    angle = np.asarray(angle).flatten()

    if zero_2_2pi:
        mod_angle = angle % (2 * np.pi)
    else:
        mod_angle = (angle + np.pi) % (2 * np.pi) - np.pi

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle