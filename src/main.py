#!/usr/bin/env python3

# -*- encoding: utf-8 -*-

import json
import os

import numpy as np
import ros_numpy
import rospy
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from sensor_msgs.msg import Image


class CNN_Node:
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "..", "cfg", "config.json")

        with open(path) as infile_h:
            self.cfg = json.load(infile_h)

        self.publish_vis = self.cfg["publish_vis"]

        config_path = self.cfg["config_path"]
        checkpoint_path = self.cfg["checkpoint_path"]
        device = self.cfg["device"]
        # Initialize the model
        self.model = init_segmentor(config_path, checkpoint_path, device=device)

        # Initialize the subscribers last or else the callback will trigger
        # when the model hasn't been created
        self.sub_rectified = rospy.Subscriber(
            self.cfg["input_topic"], Image, self.image_callback
        )
        self.pub_predictions = rospy.Publisher("/cnn_predictions", Image, queue_size=1)
        self.pub_vis = rospy.Publisher("/cnn_vis", Image, queue_size=1)

    def image_callback(self, data):
        if not data.header.seq % self.cfg["use_image_every"] == 0:
            return
        img = ros_numpy.numpify(data)
        # This assumes the model was trained using BGR imagery
        # Rotate the image so it's right side up
        # TODO Consider training on flipped imagery in the future
        if self.cfg["rotate_img"]:
            img = np.flip(img, axis=(0, 1))

        # test a single image
        result = inference_segmentor(self.model, img)

        # Rotate back the prediction and input image for visualization
        # Note this image is (1, h, w) for some reason
        if self.cfg["rotate_img"]:
            result = np.flip(result, axis=(1, 2))

        # Publish the index images
        self.pub_predictions.publish(
            ros_numpy.msgify(Image, result[0].astype(np.uint16), encoding="mono16")
        )

        # If you wish to publish the visualized results
        if self.publish_vis:
            if self.cfg["rotate_img"]:
                img = np.flip(img, axis=(0, 1))
            # Blend raw image and prediction for visualization
            draw_img = self.model.show_result(
                img,
                result,
                palette=get_palette(self.cfg["palette"]),
                show=False,
                opacity=self.cfg["vis_label_opacity"],
            )

            # Publish the overlay image
            self.pub_vis.publish(ros_numpy.msgify(Image, draw_img, encoding="bgr8"))


if __name__ == "__main__":
    rospy.init_node("mmseg", log_level=rospy.INFO)

    cnn_node = CNN_Node()

    rospy.loginfo("Started MMSegmentation node")

    rospy.spin()
