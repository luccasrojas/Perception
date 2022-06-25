
# Perception Utilities
This package offers a series of ROS services that help the robot identify faces and objects to extract information for the other tools.

**Table of Contents**

- [Perception Utilities](#perception-utilities)
- [Installation](#installation)
  * [Requirements](#requirements)
  * [Dependencies](#dependencies)
    + [Libraries](#libraries)
    + [ROS Packages](#ros-packages)
  * [Install](#install)
- [Execution](#execution)
- [Usage](#usage)
  * [Navigation Graph](#navigation-graph)
    + [Places File](#places-file)
    + [Edges File](#edges-file)
  * [Services](#services)
    + [init_recog_srv](#init_recog_srv)
    + [get_labels_srv](#get_labels_srv)
    + [look_object_srv](#look_object_srv)

- [Troubleshooting](#troubleshooting)
  * [ModuleNotFoundError](#modulenotfounderror)
    + [cv_bridge](#cv_bridge)
    + [opencv](#opencv)


# Installation
## Requirements

- Linux Ubuntu 18.04 or Ubuntu 20.04
- ROS Melodic or ROS Noetic (respectively)
- Python >= 2.7

## Dependencies
### Libraries
- [opencv][opencv]
- [cv_bridge][cv_bridge]


### ROS Packages
- [perception_msgs][navigation_msgs]


## Install

1.  Clone the repository (recommended location is ~).

```bash
  cd ~
  ```

  ```bash
  git clone https://github.com/SinfonIAUniandes/perception_utilities.git
  ```

2.  Move to the root of the workspace.

  ```bash
  cd ~/perception_utilities
  ```

3.  Build the workspace.

  ```bash
  catkin_make
  ```
  ```bash
  source devel/setup.bash
  ```

# Execution

When roscore is available run:

 ```bash
  rosrun perception_utilities main_local_object.py
  ```

# Usage

## Services

Perception_utilities offers the following services:

### init_recog_srv

+ **Description:**
This service defines if object recognition should be started/stopped.

+ **Service file:** *init_recog_srv.srv*
    + **Request**: 
		+ state (bool): start/stop object recognition (True / False, respectively).
	+ **Response**:
		+ result (string): Indicates if the service was started or not.
		
+ **Call service example:**

 ```bash
 rosservice call /perception_utilities/init_recog_srv "state: 'true'"
  ```

------------

### get_labels_srv

+ **Description:**
This service provides a list of all objects that were recognized  in a given time period.

**Note:** *This service needs to be called twice to work, once to start and once to stop.*
+ **Service file:** *get_labels_srv.srv*
    + **Request**: 
		+ start (bool): 'True' to init the service, 'False' to stop and get the result of service
	+ **Response**:
		+ answer (string): A list of all objects that were recognized in that period.
		
+ **Call service example:**

```bash
rosservice call /perception_utilities/get_labels_srv "start: 'true'"
```

------------

### look_object_srv

+ **Description:**
This service allows the robot to know if within a period of time a specific object is in the current place.

**Note:** *This service needs to be called twice to work, once to start and once to stop.*

+ **Service file:** *look_object_srv.srv*
    + **Request**: 
		+ object (string): Name of the object to want to search. (*take care caplocks*)
	+ **Response**:
		+ isThere (string): Indicates if the searching object is there or not.
        

+ **Call service example:**

 ```bash
 rosservice call /perception_utilities/look_object_srv "object: 'Mask'"
  ```


  
---------------

# Troubleshooting
Below is the solution to a series of problems that can occur when trying to use the perception_utilities.
## ModuleNotFoundError
### cv_bridge
This error occurs when  have a version problem.

To solve this problen you need to make some changes:

+ At the beginning to the perception_utilities.py
    + **If you have ROS 18.04**: Change this line  
    ```bash
    #!/usr/bin/env python3
    ```  
    to
    ```bash
    #!/usr/bin/env python
    ```  
    + **If you have ROS 20.04**: Change this line
    ```bash
    #!/usr/bin/env python
    ```  
    to
    ```bash
    #!/usr/bin/env python3
    ```  




*If the error persist, run this command:*

```bash
sudo apt-get install ros-(ROS version name)-cv-bridge
```
### Opencv

Installing opencv through terminal:

```bash
sudo apt install libopencv-dev python3-opencv
```




[Numpy]: https://numpy.org "Numpy"
[opencv]: opencv.org "opencv"
[cv_bridge]: http://wiki.ros.org/cv_bridge
[navigation_msgs]: https://github.com/SinfonIAUniandes/navigation_msgs "navigation_msgs"
[move_base]: http://wiki.ros.org/move_base "move_base"

[rospkg]: http://wiki.ros.org/rospkg "rospkg"
