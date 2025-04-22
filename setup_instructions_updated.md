# AirSim Autonomous Car Setup Instructions (Updated)

This document provides step-by-step instructions for setting up the AirSim environment, Unreal Engine, and the Python code for autonomous vehicle control on the "MountainLandscape" map.

## Prerequisites

- Windows operating system
- Python 3.7.9 installed
- Visual Studio Code installed
- AirSim v1.2.0 installed

## Step 1: Install Required Python Packages

Open a command prompt or PowerShell window and run the following commands to install the required Python packages:

```bash
pip install -r requirements.txt
```

Alternatively, you can install each package individually:

```bash
pip install airsim==1.2.0
pip install numpy==1.20.3 opencv-python==4.5.3.56 pandas==1.3.4
pip install torch==1.10.0 torchvision==0.11.1
pip install matplotlib==3.4.3 einops==0.4.1 gym==0.21.0
pip install stable-baselines3==1.5.0 pillow==8.4.0
```

## Step 2: Set Up the MountainLandscape Environment

1. **Launch the Environment**:
   - Navigate to the MountainLandscape folder you downloaded
   - Run `LandscapeMountains.exe` from the root folder (or alternatively from the `LandscapeMountains/Binaries/Win64/` folder)
   - Wait for the environment to load (this may take a few minutes)
   - Note: Since you have the packaged build version, you don't need to open a .uproject file in Unreal Engine Editor

2. **Start the Simulation**:
   - Once the environment loads, you'll see the main menu
   - Click on "Play" to start the simulation
   - You should see the SUV car spawned in the environment

## Step 3: Configure AirSim Settings

1. **Create AirSim Settings Directory**:
   - Press Win+R, type `%USERPROFILE%`, and press Enter
   - Create a folder named "Documents" if it doesn't exist
   - Inside Documents, create a folder named "AirSim" if it doesn't exist

2. **Copy Settings File**:
   - Copy the `settings.json` file from the project directory to the AirSim folder you just created
   - The path should be: `%USERPROFILE%\Documents\AirSim\settings.json`

## Step 4: Set Up the VS Code Project

1. **Create Project Directory**:
   - Create a new folder for your project (e.g., `airsim_autonomous_car`)
   - Open VS Code and select "File > Open Folder..." to open this folder

2. **Copy Project Files**:
   - Copy all the Python files from this package to your project folder:
     - `collect_data.py`
     - `airsim_env.py`
     - `run_control.py`
     - `requirements.txt`
   - Create a subfolder named `models` and copy:
     - `models/mobilevit.py`
   - Copy `settings.json` to your project root folder

3. **Verify Project Structure**:
   Your project structure should look like this:
   ```
   airsim_autonomous_car/
   ├── collect_data.py
   ├── airsim_env.py
   ├── run_control.py
   ├── settings.json
   ├── requirements.txt
   └── models/
       └── mobilevit.py
   ```

## Step 5: Running the Simulation

### Step 5.1: Start the MountainLandscape Environment

1. Launch the MountainLandscape environment by running `LandscapeMountains.exe`
2. Wait for the environment to fully load
3. Click "Play" to start the simulation
4. You should see the SUV car spawned in the environment

### Step 5.2: Collect Training Data

1. Open VS Code and navigate to your project folder
2. Open a terminal in VS Code (Terminal > New Terminal)
3. Run the data collection script:
   ```bash
   python collect_data.py
   ```
4. Use the following keyboard controls in the AirSim window to drive the car:
   - Arrow keys for steering and throttle
   - W/S for forward/backward
   - A/D for left/right steering
5. Drive around the environment to collect diverse training data
6. The script will save images and telemetry data to the `collected_data` folder
7. Press 'q' in the OpenCV window or Ctrl+C in the terminal to stop data collection

### Step 5.3: Train the Model

#### Option 1: Supervised Learning (Recommended for Initial Training)

1. In the VS Code terminal, run:
   ```bash
   python run_control.py --mode train_supervised --data_dir collected_data --model_dir saved_models --batch_size 32
   ```
2. This will train the MobileViT model using the collected data
3. The trained model will be saved to the `saved_models` directory

#### Option 2: Reinforcement Learning with SAC

1. In the VS Code terminal, run:
   ```bash
   python run_control.py --mode train_sac --model_dir saved_models --episodes 1000 --batch_size 64
   ```
2. This will train the SAC agent in the AirSim environment
3. The training process will take a significant amount of time
4. The trained model will be saved to the `saved_models` directory

### Step 5.4: Run the Autonomous Car

1. Make sure the MountainLandscape simulation is running
2. In the VS Code terminal, run:
   ```bash
   python run_control.py --mode run --model_dir saved_models --render
   ```
3. The car will now drive autonomously using the trained model
4. A window will display the center camera view with telemetry information
5. Press 'q' in the window or Ctrl+C in the terminal to stop the autonomous driving

## Troubleshooting

### Connection Issues

If you encounter connection issues between Python and AirSim:

1. Make sure the MountainLandscape simulation is running
2. Check that the settings.json file is correctly placed in:
   - %USERPROFILE%\Documents\AirSim\
3. Restart the MountainLandscape simulation
4. Restart your Python script

### Performance Issues

If you experience performance issues:

1. Reduce the resolution in settings.json (e.g., change 320x240 to 160x120)
2. Close unnecessary applications
3. Reduce graphics settings in the MountainLandscape executable (look for graphics settings in the main menu)

### Crash Recovery

If AirSim or the MountainLandscape environment crashes:

1. Save any unsaved work in VS Code
2. Restart the MountainLandscape executable
3. Resume your work from the last saved point

## Advanced Configuration

### Modifying Camera Settings

To modify camera settings, edit the `settings.json` file:

1. Change camera positions by adjusting the X, Y, Z values
2. Change camera orientations by adjusting the Pitch, Roll, Yaw values
3. Change camera resolution by modifying Width and Height values
4. Change field of view by adjusting FOV_Degrees

### Customizing Training Parameters

To customize training parameters, modify the command-line arguments:

1. For supervised learning:
   ```bash
   python run_control.py --mode train_supervised --data_dir collected_data --model_dir saved_models --batch_size 64
   ```

2. For reinforcement learning:
   ```bash
   python run_control.py --mode train_sac --model_dir saved_models --episodes 2000 --batch_size 128
   ```
