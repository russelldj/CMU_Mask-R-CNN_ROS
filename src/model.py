import init_segmentor, inference_segmentor


class Segmentor:
    def __init__(self, config_path, checkpoint_path, device):

        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(config_path, checkpoint_path, device=device)

    def forward(self, image):
        """
        Forward Prop for the Mask-R-CNN model

        Parameters:
            image (cv.Mat/np.ndarray): the input image

        Returns:
            TODO
        """

        result = inference_segmentor(self.model, image)
        return result

    def visualize(self, input_image, output):
        """
        Visualize the results of the model

        Parameters:
            image (cv.Mat/np.ndarray): the input image
            output (dict): output from the detectron2 predictor

        Returns:
            np.ndarray: visualized results
        """
        raise NotImplementedError
