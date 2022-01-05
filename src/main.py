#!/home/aaron/py36/bin/python

# -*- encoding: utf-8 -*-

import json
import os
import time

import cv2 as cv
import numpy as np
import rospy
from CMU_Mask_R_CNN.msg import predictions
from cv_converter import CV_Converter
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette


class CNN_Node:
    def __init__(self):
        # TODO update this
        path = os.path.join(os.path.dirname(__file__), "..", "cfg", "config.json")

        with open(path) as infile_h:
            self.cfg = json.load(infile_h)

        config_path = self.cfg["config_path"]
        checkpoint_path = self.cfg["checkpoint_path"]
        device = self.cfg["device"]
        self.model = init_segmentor(config_path, checkpoint_path, device=device)
        self.cv_converter = CV_Converter()

        # Initialize the subscribers last or else the callback will trigger
        # when the model hasn't been created
        self.sub_rectified = rospy.Subscriber(
            "/mapping/left/image_raw", Image, self.image_callback
        )
        self.pub_predictions = rospy.Publisher(
            "/cnn_predictions", predictions, queue_size=1
        )
        self.pub_results = rospy.Publisher("/cnn_vis", Image, queue_size=1)

    def image_callback(self, data):
        if not data.header.seq % self.cfg["use_image_every"] == 0:
            return
        cv_img = self.cv_converter.msg_to_cv(data)

        # If a vertical flip is needed
        cv_img = cv.flip(cv_img, flipCode=-1)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

        # test a single image
        result = inference_segmentor(self.model, cv_img)

        # blend raw image and prediction
        draw_img = self.model.show_result(
            cv_img,
            result,
            palette=get_palette(self.cfg["palette"]),
            show=False,
            opacity=self.cfg["opacity"],
        )

        # If you wish to publish the visualized results
        self.pub_results.publish(
            self.cv_converter.cv_to_msg(cv.flip(draw_img, flipCode=-1))
        )

        ## Convert outputs from forward call to predictions message
        # pub_msg = predictions()
        # pub_msg.header = Header(stamp=data.header.stamp)

        # pub_msg.scores = scores
        # pub_msg.bboxes = [
        #    BoundingBox2D(center=Pose2D(x=i[0], y=i[1]), size_x=i[2], size_y=i[3])
        #    for i in bboxes
        # ]
        # pub_msg.masks = [
        #    self.cv_converter.cv_to_msg(masks[i], mono=True)
        #    for i in range(masks.shape[0])
        # ]

        # pub_msg.source_image = self.cv_converter.cv_to_msg(cv.flip(cv_img, flipCode=-1))

        # self.pub_predictions.publish(pub_msg)


if __name__ == "__main__":
    rospy.init_node("cnn", log_level=rospy.INFO)

    cnn_node = CNN_Node()

    rospy.loginfo("started cnn node")

    rospy.spin()
