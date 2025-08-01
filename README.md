# SKGE-SWIN: End-To-End Autonomous Vehicle Waypoint Prediction and Navigation Using Skip Stage Swin Transformer

## Notes:
1. Some files are copied and modified from [[TransFuser, CVPR 2021]](https://github.com/autonomousvision/transfuser) repository. Please go to their repository for more details.
2. I assume you are familiar with Linux, python3, NVIDIA CUDA Toolkit, PyTorch GPU, and other necessary packages. Hence, I don't have to explain much detail.
3. Install Unreal Engine 4 and CARLA:
    - For UE4, follow: https://docs.unrealengine.com/4.27/en-US/SharingAndReleasing/Linux/BeginnerLinuxDeveloper/SettingUpAnUnrealWorkflow/
    - For CARLA, go to https://github.com/carla-simulator/carla/releases/tag/0.9.10.1 and download prebuilt CARLA + additional maps. Then, extract them to a directory (e.g., ~/OSKAR/CARLA/CARLA_0.9.10.1)

## Steps:
1. Download the dataset and extract to subfolder data. Or generate the data by yourself.
2. To train-val-test each model, go to their folder and read the instruction written in the README.md file
    - [SKGE-SWIN](https://github.com/fachrinnk4869/end-to-end-self-driving/tree/main/x13_x13_swin_torchv2) (proposed model)
    - [SKGE-SWIN](https://github.com/fachrinnk4869/end-to-end-self-driving/tree/main/x13_x13_swin_torchv2_16) (proposed model float 16)
    - [X13](https://github.com/oskarnatan/end-to-end-driving/tree/main/x13) 
    - [S13](https://github.com/oskarnatan/end-to-end-driving/tree/main/s13)
    - [CILRS](https://github.com/oskarnatan/end-to-end-driving/tree/main/cilrs)
    - [AIM](https://github.com/oskarnatan/end-to-end-driving/tree/main/aim)
    - [LF](https://github.com/oskarnatan/end-to-end-driving/tree/main/late_fusion)
    - [GF](https://github.com/oskarnatan/end-to-end-driving/tree/main/geometric_fusion)
    - [TF](https://github.com/oskarnatan/end-to-end-driving/tree/main/transfuser)

## Generate Data and Automated Driving Evaluation:
1. Run CARLA server:
    - CUDA_VISIBLE_DEVICES=0 ./CARLA/CarlaUE4.sh -opengl --world-port=2000
2. To generate data / collect data, Run expert (results are saved in subfolder 'data'):
    - CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_expert.sh
3. For automated driving, Run agents (results are saved in subfolder 'data'):
    - CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_evaluation.sh

## To do list:
1. Add download link for the dataset (The dataset is very large. I recommend you to generate the dataset by yourself :D)
