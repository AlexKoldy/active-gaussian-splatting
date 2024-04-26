from typing import Dict, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import habitat_sim


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
        
        # move quad out of scene so it doesn't show up in the images
        quad_state = self.get_quad_state()
        sim.set_agent_state(
        position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0])
        
        # Get RGBA data from the color camera
        rgba = self.sim.get_sensor_observations(0)["color_sensor"]

        # Get depth data from the depth camera
        depth = self.sim.get_sensor_observations(0)["depth_sensor"]

        # Display the image if necessary
        if display:
            rgb_img = Image.fromarray(rgba, mode="RGBA")
            plt.figure()
            plt.imshow(rgb_img)
            plt.show()

        # move quad back to original position
        self.set_quad_state(quad_state)

        # Concatenate the data and return it. The alpha value is not returned
        return np.concatenate((rgba[:, :, :-1], np.expand_dims(depth, axis=2)), axis=2)


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.getcwd(), "data")
    path_to_scene_file = os.path.join(
        data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb"
    )
    sim = Simulator(
        path_to_scene_file=path_to_scene_file, image_height=256, image_width=256
    )
    )
    rgbd = sim.collect_image_data(True)
