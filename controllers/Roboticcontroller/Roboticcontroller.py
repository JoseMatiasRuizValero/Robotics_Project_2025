from controller import Supervisor
import math  # Import the mathematics library for calculating angles
import os, sys
import numpy as np

ROOT = os.path.abspath (os.path.join(os.path.dirname(__file__),"..",".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from controllers.q_learning_agent.q_learning_agent import QLearningAgent
from utils.rewards import calculate_reward

TIMESTEP = 64
MAX_V = 6.28  # e-puck Max speed

# ===================================================================
# --- 1. Define constants, groups and thresholds --
# ===================================================================


# This is (x, z) position
GOAL_POSITION = (0.5, 0.5)  


# Corrected sensor grouping (based on e-puck standard layout)
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

# ===================================================================
# ---1. Define the robotic action set (0-based indexing)
# ===================================================================
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_LIST = [ACTION_STOP, ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT]
ACTION_SIZE = len(ACTION_LIST)
# ===================================================================
STATE_SIZE = N_SensorLevels * N_SensorLevels * N_SensorLevels * N_GoalLevels
MAX_EPISODE = 50
MAX_STEPS_PER_EPISODE = 1000

robot = Supervisor()

# 8 Infrared distance sensors: ps0..ps7
ps = [robot.getDevice(f'ps{i}') for i in range(8)]
for s in ps:
    s.enable(TIMESTEP)

# Pose: GPS + IMU (yaw)
gps = robot.getDevice('gps');
gps.enable(TIMESTEP)
imu = robot.getDevice('inertial unit');
imu.enable(TIMESTEP)

# Wheels: velocity control
lw = robot.getDevice('left wheel motor')
rw = robot.getDevice('right wheel motor')
for m in (lw, rw):
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

node = robot.getSelf()
tField = node.getField('translation')
rField = node.getField('rotation')

def reset_robot (x=0.0, z=0.0, yaw=0.0):
    node.resetPhysics()  # clear all velocities and forces
    tField.setSFVec3f([x, 0.005, z])
    rField.setSFRotation([0.0, 1.0, 0.0, float(yaw)])
    lw.setVelocity(0.0)
    rw.setVelocity(0.0)

agent = QLearningAgent(stateSize=STATE_SIZE, actionSize=ACTION_SIZE, learningRate=0.7, discountFactor=0.90, epsilon=1.0, epsilonMin=0.1, epsilonDecay=0.995)



def velocityAction(action):
    """Execute action - action is 0,1,2,3"""
    if action == ACTION_FORWARD:  # 1
        return 0.5 * MAX_V, 0.5 * MAX_V
    elif action == ACTION_LEFT:   # 2
        return -0.08 * MAX_V, 0.08 * MAX_V
    elif action == ACTION_RIGHT:  # 3
        return 0.08 * MAX_V, -0.08 * MAX_V
    elif action == ACTION_STOP:   # 0
        return 0.0, 0.0
    else:
        return 0.0, 0.0  # fallback



episode = 0
stepsInEpisode = 0
prev_state = None
prev_action = None
prev_dist = None
reset_robot(0.0, 0.0, 0.0)
robot.step(TIMESTEP)

while robot.step(TIMESTEP) !=-1:
    ps_values = [s.getValue() for s in ps]
    x, _, z = gps.getValues()  # Webots Y-axis is height, we use (x, z)
    yaw = imu.getRollPitchYaw()[2]  # Yaw angle
    
    # calculate current distance to goal
    current_dist = math.sqrt((x - GOAL_POSITION[0])**2 + (z - GOAL_POSITION[1])**2)

    val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
    val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
    val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

    if val_front < SENSOR_THRESHOLDS[0]:
        state_F = 0
    elif val_front < SENSOR_THRESHOLDS[1]:
        state_F = 1
    else:
        state_F = 2

    if val_left < SENSOR_THRESHOLDS[0]:
        state_L = 0
    elif val_left < SENSOR_THRESHOLDS[1]:
        state_L = 1
    else:
        state_L = 2

    if val_right < SENSOR_THRESHOLDS[0]:
        state_R = 0
    elif val_right < SENSOR_THRESHOLDS[1]:
        state_R = 1
    else:
        state_R = 2


    # Calculate the angle between the goal and the robot
    target_angle = math.atan2(
        GOAL_POSITION[1] - z,  # Goal z - Current z
        GOAL_POSITION[0] - x  # Goal x - Current x
    )

    # Calculate the relative angle (Target Angle - Robot Orientation)
    relative_angle = target_angle - yaw

    # Normalize to -pi to +pi (-3.14 to +3.14)
    if relative_angle > math.pi: relative_angle -= 2 * math.pi
    if relative_angle <= -math.pi: relative_angle += 2 * math.pi

    # Discretization: 0(Front), 1(Left), 2(Right)
    if abs(relative_angle) < (math.pi / 3):  # Within +/- 60 degrees (pi/3)
        state_G = 0  # Goal is in front
    elif relative_angle < 0:
        state_G = 1  # Goal is to the left (negative angle)
    else:
        state_G = 2  # Goal is to the right (positive angle)

        # ===================================================================
    # --- 4. Combine Final State ID ---
    # ===================================================================

    # We use a formula to map (L, F, R, G) to a unique number
    # (L * 27) + (F * 9) + (R * 3) + G
    StateID = (state_L * (N_SensorLevels * N_SensorLevels * N_GoalLevels)) + \
              (state_F * (N_SensorLevels * N_GoalLevels)) + \
              (state_R * N_GoalLevels) + \
              state_G

    StateID = int(StateID)  # Ensure it is an integer

    # ===================================================================

    # Print the calculated state for debugging
    print(f"L:{state_L} F:{state_F} R:{state_R} G:{state_G}  --->  StateID: {StateID}")

    # Print raw values to help adjust SENSOR_THRESHOLDS
    # print(f"Raw: L={val_left:.0f} F={val_front:.0f} R={val_right:.0f} Ang={relative_angle:.2f}")

    # ===================================================================
    # --- 5 .action choose(from keyboard) ---
    # ===================================================================

    #q learning choose action
    actionIdx = agent.chooseAction(StateID)
    chosenAction = ACTION_LIST[actionIdx]

    #execute action
    vL, vR = velocityAction(chosenAction)
    lw.setVelocity(vL)
    rw.setVelocity(vR)

    # compute reward
    reward, done = calculate_reward(robot_pos=(x, z), goal_pos=GOAL_POSITION, sensors=[val_left, val_front, val_right],
                                    action=chosenAction, prev_distance=prev_dist
                                    )
    
    # update prev_dist for next iteration
    prev_dist = current_dist

    if prev_state is not None:
        agent.update(prev_state, prev_action, reward, StateID)

    prev_state = StateID
    prev_action = actionIdx
    stepsInEpisode += 1

    if done or stepsInEpisode >= MAX_STEPS_PER_EPISODE:
        episode += 1
        print(f"episode {episode} finished | steps={stepsInEpisode} | last reward={reward}")
        agent.endEpisode()
        agent.save(os.path.join(RESULTS_DIR, "q_table.npy"))

        lw.setVelocity(0.0)
        rw.setVelocity(0.0)

        prev_state = None
        prev_action = None
        prev_dist = None
        stepsInEpisode = 0
        reset_robot(0.0,0.0,0.0)
        robot.step(TIMESTEP)

        if episode >=MAX_EPISODE:
            print("training finished")
            break
        continue
