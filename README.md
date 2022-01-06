# MMSegmentation Predictions
This is a `ros` node that publishes predictions from an `mmsegmentation` segmentation.

# Setup
You need to create an environment that contains `rospkg` and the necessary dependencies for `mmsegmentation`. An example conda file is `mmseg_ros.env`. Alternatively, you can follow the directions from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) directly. Finally, you need to clone my fork of [`mmsegmentation`](https://github.com/russelldj/mmsegmentation?organization=russelldj&organization=russelldj) and install it by `cd`ing to the repo location and running `pip install -e .` (note the trailing period).

Make a directory called `catkin_ws` and run `catkin_make`. Go to the `src` dir and clone this repo. Also in the `src` dir clone [ros_numpy](git@github.com:eric-wieser/ros_numpy.git). Rerun `catkin_make`. Source `devel/setup.{sh|zsh}` to set up the environment. Run `rosrun mmsegmentation_predictions main.py` to start the node.

# Configuring
You can edit most imporant aspects in the `cfg/config.json` file.

# Topics
By default this node listens on `/mapping/left/image_color` but this can be changed in the `input_topic` field of the `cfg/config.json` file. It publishes a visualization image on `/seg_vis` which is the classes overlaid on a grayscale version of the image. The class indices are published on `/seg_class_predictions`.

# Models
Models are hosted [in this private repo](https://github.com/russelldj/SafeForestData). Please contact me if you think you should have access to it. You can use any model you wish though, just update `config_path` and `checkpoint_path` in `cfg/config.json`.

# License
Distributed under BSD-3.