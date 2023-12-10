import numpy as np

class DroneData:
    def __init__(self) -> None:
        self.id : int = None
        self.position : np.ndarray = None
        self.angle : float = None
        self.wounded_found : list = None
        self.wounded_target : int = None
