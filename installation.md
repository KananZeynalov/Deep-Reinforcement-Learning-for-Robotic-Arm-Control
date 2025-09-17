# **Installation Guide for FetchReach Environment on Gymnasium**
This document provides a step-by-step guide to install and configure the FetchReach environment from Gymnasium Robotics on Ubuntu 22.04 LTS using Python 3.10.12. In addition to Python dependencies (listed in requirements.txt), you will need to manually install and configure MuJoCo. Please follow the instructions exactly to ensure proper functionality.

Installing gymnasium is a very tedious task where it both requires many packages to install both by pip and apt
## **1. Prerequisites (IMPORTANT)**
Before proceeding, make sure you have the following:

Operating System: Ubuntu 22.04 LTS

Python: Version 3.8–3.10 (this guide uses Python 3.10.12)

pip: Latest version of Python package installer

Cython: Version less than 3 (required for building mujoco-py)


### 2. Install System Dependencies**

Run these commands in your terminal to update your system and install the required system libraries:

```bash
sudo apt update
sudo apt install -y libgl1-mesa-dev libosmesa6-dev libglew-dev patchelf
sudo apt install -y python3.10 python3.10-venv python3-pip
```

## 3. Install Python Dependencies and Activate a Virtual Environment

All of necessary dependencies are in requirements.txt, so you need virtual environment to avoid any *VERSION* mismatch. Therefore proceed with below:

1. Either create virtual environment(venv) with below: 
```bash
python3.10 -m venv test
source test/bin/activate
```

After venv created proceed with below for Python packages:

```bash
pip install -r requirements.txt
```


2. *(OPTIONAL)* You can create Conda environment for your comfort but it is a long process therefore you should be aware of it yourself as it is long. Therefore recommending above approach with venv.


## 4. Mujoco setup (extra careful)



Due to Ubuntu’s need for binary files, you must download and configure MuJoCo manually. Preferred approach is to download in this opened terminal /project/folder and then proceed with it 

https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

```bash
mkdir -p ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Set the environment variables. (IT SHOULD BE EXACTLY SAME PLEASE DO IN THIS TERMINAL)
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export MUJOCO_GL=egl

source ~/.bashrc
```


## Checkup for RUN

Verify the code by running prerun.py to see if robotic environment is working. You will see a robot trying to catch bal RANDOMLY. Move forward only if it works. If it does not work it means you have not setup mujoco correctly. For debug you can do *ls-ali* to see if mujoco is correctly installed and extracted in ~/.

## VERY IMPORTANT GYMNASIUM SETUP
Due to a known bug in the gymnasium-robotics implementation, you need to modify the robot initial positions in the FetchReach environment:

Navigate to the fetch/reach.py file. The typical path is:

YOUR_VENV/lib/python3.10/site-packages/gymnasium_robotics/envs/fetch/reach.py

Open the file and locate line 124 (or the corresponding section where the robot’s initial joint positions are set).

Change the initialization dictionary to:

```python
initial_qpos = {
    "robot0:slide0": -0.2,
    "robot0:slide1": 0.20,
    "robot0:slide2": 0.0,
    }
```
This change ensures the target is reachable during simulation.

## Possible Problems & Solutions

Now you can read README.md after this 

##  POSSIBLE ONE ISSUE

Issue: Cython Errors (e.g., ‘noexcept’ deprecated)
If you encounter *Cython* errors like:
```csharp
error: invalid argument '-std=c++14' not allowed with 'C'
or references to “noexcept,” install a lower Cython version with pip below:
```
```bash
pip install "Cython<3"
```
