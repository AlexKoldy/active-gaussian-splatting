from typing import Dict, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import habitat_sim
from scipy.spatial.transform import Rotation as R
import os


class Simulator:
    """
    Habitat Simulator for active perception using Guassian Splatting
    """

    def __init__(
        self, path_to_scene_file: str, image_height: int = 256, image_width: int = 256
    ) -> None:
        """
        Default simulator constructor

        Arguments:
            path_to_scene_file (str): full path to the scene data file
            image_height (int): spatial height resolution of observation
            image_width (int): spatial width resoltuion of observation
        """
        # Set up simulation settings
        settings = {
            "scene": path_to_scene_file,
            "default_agent": 0,
            "sensor_height": 0.0,
            "height": image_height,
            "width": image_width,
        }

        # Set up simulation environment and initialize agent
        config = self.make_simple_config(settings)
        self.sim = habitat_sim.Simulator(config)
        self.agent = self.sim.initialize_agent(settings["default_agent"])

        # self.sim.pathfinder.load_nav_mesh(
        #     "/home/alko/ese6500/active-gaussian-splatting/data/versioned_data/habitat_test_scenes/apartment_1.navmesh",
        # )

    def make_simple_config(
        self, settings: Dict[str, Union[str, float, int]]
    ) -> habitat_sim.Configuration:
        """
        Generates a configuration for the simulator

        Arguments:
            settings (Dict[str, Union[str, float, int]]): simulator settings

        Returns:
            (habitat_sim.Configuration): simulator configuration object
        """
        # Set up simulator
        sim_config = habitat_sim.SimulatorConfiguration()
        sim_config.scene_id = settings["scene"]

        # Set up agent
        agent_config = habitat_sim.agent.AgentConfiguration()

        # Set up RGB camera
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

        # Set up depth camera
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

        # Attach sensors to agent
        agent_config.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

        # Return a full configuration containing the simulator and agent
        return habitat_sim.Configuration(sim_config, [agent_config])

    def set_agent_state(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """
        Sets the agent's position and orientation in the simulator enviroment

        Arguments:
            position (np.ndarray): global (x,y,z)-position of the agent [m]
            orientation (np.ndarray): (x,y,z,w)-quaternion orientation of the agent
        """
        # Set up agent state
        agent_state = habitat_sim.AgentState()

        # Set the states as floats, as integer arrays will throw errors
        agent_state.position = position.astype(np.float64, copy=False)
        agent_state.rotation = orientation.astype(np.float64, copy=False)
        self.agent.set_state(agent_state)

    def collect_image_data(self, display: bool = False) -> np.ndarray:
        """
        Returns the RGB and depth information at each pixel

        Arguments:
            display (bool): flag for displaying the image to the user. Default is 'False'

        Returns:
            (np.ndarray): RGB, depth values as numpy array of shape (height, width, 4)
        """

        # # move quad out of scene so it doesn't show up in the images
        # quad_state = self.get_quad_state()
        # self.set_agent_state(position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))

        # Get RGBA data from the color camera
        rgba = self.sim.get_sensor_observations(0)["color_sensor"]

        # Get depth data from the depth camera
        depth = self.sim.get_sensor_observations(0)["depth_sensor"]

        # Display the image if necessary
        if display:
            rgb_img = Image.fromarray(rgba, mode="RGBA")
            plt.figure()
            plt.imshow(rgb_img)
            # plt.show()
            plt.savefig("test.png")
            plt.close()

        # move quad back to original position
        # self.set_quad_state(quad_state)

        # Concatenate the data and return it. The alpha value is not returned
        return np.concatenate((rgba[:, :, :-1], np.expand_dims(depth, axis=2)), axis=2)

    def sample_images_from_poses(self, poses):
        """
        sample images from list of poses

        Args:
            poses: list of numpy arrays of pose (x, y, z, qx, qy, qz, qw)

        Returns:
            list of images
        """
        # move quad out of scene so it doesn't show up in the images
        # quad_state = self.agent.get_state()

        rgbs = []
        depths = []
        for pose in poses:
            if pose.ndim < 2:
                orientation = R.from_euler("zyx", pose[3:]).as_quat()
                position = pose[:3]
            else:
                orientation = R.from_matrix(pose[:3, :3]).as_quat()
                position = pose[:3, 3]
            self.set_agent_state(position, orientation)
            rgbd = self.collect_image_data()
            # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgbs.append(rgbd[:, :, :-1])
            depths.append(rgbd[:, :, -1])

        # move quad back to original position
        # self.set_agent_state(quad_state)
        return np.array(rgbs), np.array(depths)

    # convert 3d points to 2d topdown coordinates
    def convert_points_to_topdown(self, points, meters_per_pixel):
        points_topdown = []
        bounds = self.sim.pathfinder.get_bounds()
        for point in points:
            # convert 3D x,z to topdown x,y
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return np.array(points_topdown).T


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.getcwd(), "data")
    path_to_scene_file = os.path.join(
        data_path, "versioned_data/habitat_test_scenes/apartment_1.glb"
    )

    sim = Simulator(
        # path_to_scene_file=path_to_scene_file,
        path_to_scene_file="/home/alko/ese6500/active-gaussian-splatting/data/versioned_data/habitat_test_scenes/apartment_1.glb",
        image_height=256,
        image_width=256,
    )
    # https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/notebooks/ECCV_2020_Navigation.ipynb

    height = sim.sim.pathfinder.get_bounds()[0][1]
    # height = 0

    map = sim.sim.pathfinder.get_topdown_view(0.01, height)
    print(sim.sim.pathfinder.is_loaded)

    waypoints = np.array([[0, 0, 0], [0, 0, 1]])

    points = sim.convert_points_to_topdown(waypoints, 0.1)

    # print(map)
    # map = np.ones((256, 256))

    # print(type(map))
    # plt.figure()
    # plt.imshow(map)
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(points[0, 0], points[1, 0])
    plt.plot(points[0, :], points[1, :])
    plt.imshow(map)
    plt.savefig("TOPDOWN.png")
    plt.close()

    quat = np.array([0, -0.7068252, 0, 0.7068252])

    sim.set_agent_state(np.array([2.0, 0, 0]), quat / np.linalg.norm(quat))
    rgbd = sim.collect_image_data(True)
