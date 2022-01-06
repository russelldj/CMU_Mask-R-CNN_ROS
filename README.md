# MMSegmentation Predictions
This is a `ros` node that publishes predictions from an `mmsegmentation` segmentation.

# Setup
Make a directory called `catkin_ws`. Run `catkin_make`. Go to the `src` dir and clone this repo. Also in the `src` dir clone [vision_msgs](https://github.com/ros-perception/vision_msgs). Rerun `catkin_make`. Source `devel/setup.{sh|zsh}`. Run `rosrun CMU_Mask_R_CNN main.py`.

# Configuring
You can edit most imporant aspects in the `cfg/config.json` file.

# Topics
By default this node listens on `/mapping/left/image_color` but this can be changed in the `input_topic` field of the `cfg/config.json` file. It publishes a visualization image on `/seg_vis` which is the classes overlaid on a grayscale version of the image. The class indices are published on `/seg_class_predictions`.

# Models
Models are hosted [in this private repo](https://github.com/russelldj/SafeForestData). Please contact me if you think you should have access to it. 

# License
Distributed under BSD