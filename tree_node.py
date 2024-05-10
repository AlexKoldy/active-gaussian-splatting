import numpy as np


class TreeNode:
    def __init__(
        self,
        x: float,
        y: float,
        parent: "TreeNode" = None,
        cost: float = None,
        is_initial: bool = False,
    ) -> None:
        """ """
        self.x = x
        self.y = y

        self.parent = parent
        self.cost = cost

        self.is_root = is_initial

    def get_position(self) -> np.ndarray:
        """ """
        return np.array([self.x, self.y])

    def set_parent(self, parent: "TreeNode") -> None:
        """ """
        self.parent = parent
