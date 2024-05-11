import numpy as np
from typing import List, Tuple

from tree_node import TreeNode


class RapidlyExploringRandomTreePlanner:
    def __init__(
        self,
        gaussModel,
        move_distance: float,
        k: int,
        z: float,
        num_points_to_check: int,
        cost_collision_thresh: float,
        max_samples: int,
        goal_tresh: float,
        bounds: np.ndarray = None,
    ) -> None:
        """ """

        self.gaussModel = gaussModel
        # Set planning parameters
        self.move_distance = move_distance
        self.k = k
        self.z = z
        self.num_points_to_check = num_points_to_check
        self.cost_collision_thresh = cost_collision_thresh
        self.max_samples = max_samples
        self.goal_tresh = goal_tresh

        # Set bounds of enviornment
        self.set_bounds(bounds)

    def set_bounds(self, bounds: np.ndarray) -> None:
        """ """
        self.bounds = bounds

    def sample(self) -> np.ndarray:
        """"""
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def find_nearest_node_to_point(
        self, tree: List[TreeNode], point: np.ndarray
    ) -> TreeNode:
        """ """
        nearest_node = None
        nearest_distance = float("inf")
        for node in tree:
            distance = np.linalg.norm(node.get_position() - point)
            if distance < nearest_distance:
                nearest_node = node
                nearest_distance = distance

        return nearest_node

    def move_towards_point(self, nearest_node: TreeNode, point: np.ndarray) -> TreeNode:
        """ """
        # Get the normalized direction vector from the closest node to
        # the sampled point
        direction = point - nearest_node.get_position()
        direction = direction / np.linalg.norm(direction)

        # Calculate the position of the new node
        new_position = nearest_node.get_position() + self.move_distance * direction

        # Generate a new potential node for the tree
        new_node = TreeNode(new_position[0], new_position[1], nearest_node)

        return new_node

    def find_k_closest_gaussians(
        self, point: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        # Get the position and covariance of the Gaussians. The covariance is
        # represented as a vector of size 6 holding the upper triangle of the
        # symmetric 3x3 covariance matrix Ensure the xyz positions of the Gaussian
        # models is of shape (3, N)
        mus = (
            self.gaussModel.get_xyz.T.detach().cpu().numpy()
        )  # Positions of shape (3, N)
        Sigmas = (
            self.gaussModel.get_covariance().detach().cpu().numpy()
        )  # Covariances of shape (N, 6)

        # Calculate distances
        point = np.expand_dims(point, axis=1)
        distances = np.linalg.norm(mus - point, axis=1)

        # Sort distances and get indices of k smallest ones
        indices = np.argsort(distances)[:k]

        # Return position and covariance of the k closest Gaussians
        return (mus[:, indices], Sigmas[indices, :])

    def evaluate_gaussian_at_point(
        self, mu: np.ndarray, Sigma: np.ndarray, point: np.ndarray
    ) -> float:
        """"""
        point = np.array(point)
        mu = np.array(mu)
        return (
            (2 * np.pi) ** (-3 / 2)
            * np.linalg.det(Sigma) ** (-1 / 2)
            * np.exp(-0.5 * (point - mu).T @ np.linalg.inv(Sigma) @ (point - mu))
        )

    def generate_symmetric_covariance_from_upper_triangle(
        self, upper_triangle: np.ndarray
    ) -> np.ndarray:
        """"""
        # Initialize a symmetric matrix
        symmetric_matrix = np.zeros((3, 3))

        # Fill the upper triangle
        symmetric_matrix[np.triu_indices(3, k=0)] = upper_triangle

        # Fill the lower triangle by transposing the upper triangle
        symmetric_matrix = (
            symmetric_matrix + symmetric_matrix.T - np.diag(np.diag(symmetric_matrix))
        )
        return symmetric_matrix

    # TODO: JIT this???
    # @njit
    def check_collision(self, start_point: np.ndarray, end_point: np.ndarray) -> bool:
        """ """
        # Generate a set of points along the line between the start and end points
        local_path = np.vstack(
            (
                np.linspace(start_point[0], end_point[0], self.num_points_to_check),
                np.zeros(self.num_points_to_check),
                np.linspace(start_point[2], end_point[2], self.num_points_to_check),
            )
        )

        # Set up total cost
        cost = 0

        # Loop through and add Gaussian values
        for point in local_path.T:
            # Find the k closest Gaussians
            mus, Sigmas = self.find_k_closest_gaussians(point, self.k)

            # Transpose 'mus' so we take the rows, i.e., positions of the Gaussians
            for mu, Sigma in zip(mus.T, Sigmas):
                # Add cost based off Gaussian
                cost += self.evaluate_gaussian_at_point(
                    mu,
                    self.generate_symmetric_covariance_from_upper_triangle(Sigma),
                    point,
                )

            # Check if the cost exceeds the threshold in the look to return early if needed
            if cost >= self.cost_collision_thresh:
                return True

        return False

    def step(self, root_node: TreeNode, tree: List[TreeNode]) -> None:
        """ """
        # Sample a new point in free space
        point = self.sample()

        # Find the nearest node to this point
        nearest_node = self.find_nearest_node_to_point(tree, point)

        # Move towards this node and create a new node
        new_node = self.move_towards_point(nearest_node, point)

        # Check if there is a collision based on the Gaussian cost
        # Note that we do this check in 3D space
        if not self.check_collision(
            np.array([root_node.x, self.z, root_node.y]),
            np.array([new_node.x, self.z, new_node.y]),
        ):
            new_node.set_parent(nearest_node)
            tree.append(new_node)
        else:
            print("Collision detected!")

        # TODO COMMENT OUT
        # new_node.set_parent(nearest_node)
        # tree.append(new_node)

    def is_close_to_goal(self, node: TreeNode, goal_point: np.ndarray) -> bool:
        """ """
        # Calculate the node's distance to the goal
        distance = np.linalg.norm(goal_point - node.get_position())

        # Return 'True' if close enough
        if distance > self.goal_tresh:
            return False
        else:
            return True

    def plan(self, current_point: np.ndarray, goal_point: np.ndarray) -> np.ndarray:
        """ """
        # Set up the tree with the root node at the current position
        root_node = TreeNode(
            x=current_point[0],
            y=current_point[2],
            is_initial=True,
        )
        tree = [root_node]

        # Plan while the tree has not reached the maximum number
        # of samples allowed and while the newest node is not
        # close enough to the goal
        while len(tree) <= self.max_samples:
            self.step(root_node, tree)
            if self.is_close_to_goal(tree[-1], goal_point):
                goal_node = TreeNode(x=goal_point[0], y=goal_point[1])
                goal_node.set_parent(tree[-1])
                tree.append(goal_node)
                print("RRT converged")
                break

        # Trace backwards on the tree to get the path
        path = [tree[-1]]
        while path[0] != tree[0]:
            path.insert(0, path[0].parent)
        # path.reverse()

        path_arr = np.zeros((2, len(path)))
        for i, node in enumerate(path):
            path_arr[0, i] = node.x
            path_arr[1, i] = node.y

        # path_arr = path_arr

        length = path_arr.shape[1]
        path_arr = np.vstack((path_arr[0], np.zeros(length), path_arr[1]))
        return path_arr.T
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.scatter(path_arr[0, :], path_arr[1, :], color="blue")
        # plt.plot(path_arr[0, :], path_arr[1, :], color="blue")
        # plt.scatter(current_point[0], current_point[1], marker="x", color="green")
        # plt.scatter(goal_point[0], goal_point[1], marker="x", color="red")
        # plt.show()


if __name__ == "__main__":
    rrt_planner = RapidlyExploringRandomTreePlanner(
        None,
        move_distance=0.25,  # how far to move in direction of sampled point
        k=1,  # number of Gaussians
        z=0.0,  # height of planning [m]
        num_points_to_check=10,  # number of points to check for collision
        cost_collision_thresh=100.0,  # cost threshold on whether or not there is a collision based off total sampled cost across the lin
        max_samples=1000,  # maximum number of times to try sampling a new node
        goal_tresh=0.3,  # distance from node to goal point to be considered converged
        bounds=np.array(
            [
                [-3.0, 3.0],
                [-3.0, 3.0],
            ]
        ),  # bounds in 2D space for sampling
    )

    a = np.array([0, 0])
    b = np.array([1, 2])
    rrt_planner.plan(a, b)
