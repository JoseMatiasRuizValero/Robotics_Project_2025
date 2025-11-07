from controller import Robot
import math  # Import the mathematics library for calculating angles

TIMESTEP = 64
MAX_V = 6.28  # e-puck Max speed

# ===================================================================
# --- 1. Define constants, groups and thresholds --
# ===================================================================


# This is (x, z) position
GOAL_POSITION = (0.8, 0.8)  

# !! Note: This is the assumed sensor grouping. Please verify it according to the e-puck documentation !!
# (e-puck The sensor is usually ps0 on the right front and ps7 on the left front when moving clockwise)
PS_GROUP_FRONT = [0, 7]      # sensor ps0, ps7
PS_GROUP_LEFT  = [5, 6]      # sensor ps5, ps6
PS_GROUP_RIGHT = [1, 2]      # sensor ps1, ps2


# The value of the e-puck sensor is usually between 0 (far) and approximately 4000 (near).

# We divide it into three levels: 0 (Safe), 1 (Warning), and 2 (dangerous)

SENSOR_THRESHOLDS = [
    100.0,  # Level 0: < 100 (Safe/Far away)
    500.0   # Level 1: 100-500 (Warning/Medium)
            # Level 2: > 500 (Dangerous/Very close)
]

# The number of discretization levels
N_SensorLevels = 3 # (0, 1, 2)
N_GoalLevels   = 3 # (0, 1, 2)

# ===================================================================

robot = Robot()

# 8 Infrared distance sensors: ps0..ps7
ps = [robot.getDevice(f'ps{i}') for i in range(8)]
for s in ps:
    s.enable(TIMESTEP)

# Pose: GPS + IMU (yaw)
gps = robot.getDevice('gps'); gps.enable(TIMESTEP)
imu = robot.getDevice('inertial unit'); imu.enable(TIMESTEP)

# Wheels: velocity control
lw = robot.getDevice('left wheel motor')
rw = robot.getDevice('right wheel motor')
for m in (lw, rw):
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

while robot.step(TIMESTEP) != -1:
    
    # 1. Get sensor readings (from your code)
    ps_values = [s.getValue() for s in ps]
    x, _, z = gps.getValues() # Webots Y-axis is height, we use (x, z)
    yaw = imu.getRollPitchYaw()[2] # Yaw angle

    # ===================================================================
    # --- (New) 2. Sensor Discretization (Left, Front, Right) --- 
    # ===================================================================
    
    # Aggregation: Take the maximum value in that direction (represents the closest obstacle)
    val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
    val_left  = max(ps_values[i] for i in PS_GROUP_LEFT)
    val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

    # Discretization (Binning): 0=Safe, 1=Warning, 2=Dangerous
    if val_front < SENSOR_THRESHOLDS[0]: state_F = 0
    elif val_front < SENSOR_THRESHOLDS[1]: state_F = 1
    else: state_F = 2
    
    if val_left < SENSOR_THRESHOLDS[0]: state_L = 0
    elif val_left < SENSOR_THRESHOLDS[1]: state_L = 1
    else: state_L = 2

    if val_right < SENSOR_THRESHOLDS[0]: state_R = 0
    elif val_right < SENSOR_THRESHOLDS[1]: state_R = 1
    else: state_R = 2

    # ===================================================================
    # --- (New) 3. Goal Direction Discretization --- 
    # ===================================================================
    
    # Calculate the angle between the goal and the robot
    target_angle = math.atan2(
        GOAL_POSITION[1] - z,  # Goal z - Current z
        GOAL_POSITION[0] - x   # Goal x - Current x
    )
    
    # Calculate the relative angle (Target Angle - Robot Orientation)
    relative_angle = target_angle - yaw
    
    # Normalize to -pi to +pi (-3.14 to +3.14)
    if relative_angle > math.pi: relative_angle -= 2 * math.pi
    if relative_angle <= -math.pi: relative_angle += 2 * math.pi

    # Discretization: 0(Front), 1(Left), 2(Right)
    if abs(relative_angle) < (math.pi / 3): # Within +/- 60 degrees (pi/3)
        state_G = 0 # Goal is in front
    elif relative_angle < 0:
        state_G = 1 # Goal is to the left (negative angle)
    else:
        state_G = 2 # Goal is to the right (positive angle)

    # ===================================================================
    # --- (New) 4. Combine Final State ID ---
    # ===================================================================
    
    # We use a formula to map (L, F, R, G) to a unique number
    # (L * 3*3*3) + (F * 3*3) + (R * 3) + G
    # (L * 27) + (F * 9) + (R * 3) + G
    StateID = (state_L * (N_SensorLevels * N_SensorLevels * N_GoalLevels)) + \
              (state_F * (N_SensorLevels * N_GoalLevels)) + \
              (state_R * N_GoalLevels) + \
              state_G
              
    StateID = int(StateID) # Ensure it is an integer

    # ===================================================================
    
    # Print the calculated state for debugging
    print(f"L:{state_L} F:{state_F} R:{state_R} G:{state_G}  --->  StateID: {StateID}")
    
    # Print raw values to help adjust SENSOR_THRESHOLDS
    print(f"Raw: L={val_left:.0f} F={val_front:.0f} R={val_right:.0f} Ang={relative_angle:.2f}")

    # For now, let the robot move forward at a constant speed to verify state changes
    # Later, this will be Yin-Chu and Jose's Q-Learning/SARSA algorithm
    # The algorithm will determine vL and vR based on StateID
    vL = 0.5 * MAX_V
    vR = 0.5 * MAX_V
    
    lw.setVelocity(vL)
    rw.setVelocity(vR)