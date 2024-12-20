import sys
import numpy as np
from copy import deepcopy
from math import pi
from scipy.stats import zscore

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
import time

from lib.calculateFK import FK
from lib.IK_position_null import IK
from labs.final.final_trajectory_planner import finalTrajectoryPlanner
from lib.calcAngDiff import calcAngDiff

fk = FK()
ik = IK()

LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
CENTER = LOWER + (UPPER - LOWER) / 2

H_BASE_WORLD = None
H_WORLD_BASE = None

ABOVE_DIST = 0.08

NUMBER_OF_DETECTIONS = 10

BLOCK_X_OFFSET = 0
BLOCK_Y_OFFSET = 0
BLOCK_Z_OFFSET = 0

GRIPPER_CLOSED_WIDTH = 0.045#0.04
GRIPPER_OPEN_WIDTH = 0.065

ROTATATING_TABLE_HEIGHT = 0.22   # Change to 0.12 for lab

DYNAMIC_COUNT = 0


def align_pose_with_axis(pose, axis):
    rotate_90_about_z_matrix = np.array([[0,-1,0, 0],
                                    [1,0,0, 0],
                                    [0,0,1, 0],
                                    [0,0,0,1]])
    smallest_pose = deepcopy(pose)
    for _ in range(4):
        if np.linalg.norm(pose[:,0] - axis) < np.linalg.norm(smallest_pose[:,0] - axis):
            smallest_pose = pose
        pose = np.dot(rotate_90_about_z_matrix, pose)
    return smallest_pose


def detect_blocks_with_noise(detector):
    """detects blocks multiple times and filters noise"""

    #creates a dict where each of the detected poses fits into the 
    output = []
    readings = {}
    for _ in range(NUMBER_OF_DETECTIONS):
        blocks = detector.get_detections()
        for (name, pose) in blocks:
            if name in readings:
                readings[name].append(pose)
                break
            readings[name] = [pose]

    for name, detections in readings.items():
        detections = np.array(detections)
        detections = detections.reshape((detections.shape[0], detections.shape[1]*detections.shape[2]))
        z_scores = np.abs(zscore(detections, axis=0))  # Axis=0 for column-wise z-scores
        is_outlier = z_scores > 2
        detections[is_outlier] = np.nan

        # Compute the mean along the first axis (excluding NaN values)
        filtered_output = np.nanmean(detections, axis=0)
        output.append((name, filtered_output.reshape((4,4))))
            
    return output


def detect_blocks_in_base_frame(q, detector):
    """returns the blocks name and positions where the position has z"""
    # get the transform from camera to panda_end_effector
    H_camera_ee = detector.get_H_ee_camera()
    _, H_ee_base = fk.forward(q)
    H_camera_base = H_ee_base @ H_camera_ee

    #get blocks
    detections = detect_blocks_with_noise(detector)

    #rearrange each blocks orientation so their z is pointing up
    for index, (name, pose) in enumerate(detections):
        #convert to base coords
        pose = H_camera_base @ pose

        #setup arrays
        pose = np.array(pose)
        new_pose = np.zeros(shape=(4,3))
        #find how many times to swap the columns
        for i in range(3):
            #using 0.90 as a buffer for noise, ideally the z value would be 1
            if abs(pose[2,i]) > 0.90:
                break
        columns_to_move = 2-i
        
        #swap x,y,x and z so that block z is parrallel to base z
        for old_column in range(3):
            new_column = (old_column+columns_to_move)%3
            new_pose[:, new_column] = pose[:, old_column]

        #rotate 180 degrees about x if z is pointing up
        if new_pose[2,2] > 0:
            new_pose[:,1:3] = -1 * new_pose[:,1:3]

        #rotate about z until block x is mostly facing base x
        new_pose = align_pose_with_axis(new_pose, np.array([1,0,0,0]))

        #combine with linear dist info and return to the array
        detections[index] = (name, np.hstack((new_pose, np.array(pose[:, 3]).reshape((4,1)))))

    return detections


def predict_future_location(table_omega, time_in_future, pose):
    angle_change = table_omega * time_in_future
    block_in_world = H_BASE_WORLD @ pose


    homogenous_table_transform = np.array([[np.cos(angle_change),-np.sin(angle_change),0,0],
                                           [np.sin(angle_change),np.cos(angle_change),0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]])
    
    future_block_location_in_world = homogenous_table_transform @ block_in_world

    return H_WORLD_BASE @ future_block_location_in_world

def pick_up_dynamic_block(dynamic_neutral, detector, team):
    arm.safe_move_to_position(dynamic_neutral)

    print("Finding Blocks")
    for _ in range(25):
        # Detect blocks twice with a specified delay between the readings
        blocks1 = detect_blocks_in_base_frame(dynamic_neutral, detector)
        detection_time_1 = time_in_seconds()

        if len(blocks1) == 0:
            continue

        furthest_away_block = blocks1[0]
        for name, pos in blocks1:
            if team == 'red':
                if pos[0][3] < furthest_away_block[1][0][3]:
                    furthest_away_block = (name, pos)
            else:
                if pos[0][3] > furthest_away_block[1][0][3]:
                    furthest_away_block = (name, pos)
        
        time.sleep(3)
        blocks2 = detect_blocks_in_base_frame(dynamic_neutral, detector)
        detection_time_2 = time_in_seconds()
        break

    time_diff = detection_time_1 - detection_time_2

    block_name_1, block_pos_1 = furthest_away_block

    for (name, pose) in blocks2:
        if name == block_name_1:
            block_pos_2 = pose
            break
    else:
        return False

    # Find matching blocks and predict the future position
    table_omega = -1 * np.arcsin(np.linalg.norm(calcAngDiff(block_pos_2[0:3,0:3], block_pos_1[0:3,0:3]))) / time_diff
    print(f"Table Omega: {table_omega}")
    if abs(table_omega) > 0.5:
        print("Found Omega is too large")
        return False

    # Predict future position
    time_in_future = 10
    first_pose = predict_future_location(table_omega, time_in_future - 2, pose)
    second_pose = predict_future_location(table_omega, time_in_future, pose)

    arm.exec_gripper_cmd(0.08)

    first_pose = np.hstack((align_pose_with_axis(first_pose[:,0:3], np.array([1,0,0,0])),first_pose[:,3].reshape(4,1)))
    second_pose = np.hstack((align_pose_with_axis(second_pose[:,0:3], np.array([1,0,0,0])),second_pose[:,3].reshape(4,1)))


    first_pose[2,3] = first_pose[2,3] + ABOVE_DIST
    q_above, _, success, message = ik.inverse(deepcopy(first_pose), deepcopy(goal_neutral), method='J_pseudo', alpha=0.5)
    if not success:
        print("IK could not be solved! " + message)
        return False
    arm.safe_move_to_position(q_above)

    q_target, _, success, message = ik.inverse(deepcopy(second_pose), deepcopy(q_above), method='J_pseudo', alpha=0.5)
    if not success:
        print("IK could not be solved! " + message)
        return False

    time_wait_to_close = time_in_future - (time_in_seconds() - detection_time_2) - 1
    time.sleep(time_wait_to_close)

    arm.safe_move_to_position(q_target)
    arm.exec_gripper_cmd(GRIPPER_CLOSED_WIDTH)
    arm.safe_move_to_position(dynamic_neutral)

    return True


def look_for_static_blocks_and_stack(start_neutral, tower_location, goal_pos) -> bool:
    """collects and stacks a single static block
    Args:
        start_neutral: 1x7 np array with the joint values for a configuration to see where
            the static blocks are located
        tower_location: 4x4 np array which is the homogenous transformation from where the
            block should be placed in the base frame
        start_pos: 1x3 np array with the x,y,z of the starting_netural position in base frame
        goal_pos: 1x3 np array with the x,y,z of the goal_netural position in base frame
            
    return:
        True if a block was found and placed
        False if no blocks were found"""
    arm.safe_move_to_position(start_neutral)

    #define locations in space
    blocks = detect_blocks_in_base_frame(start_neutral, detector)
    if len(blocks) == 0:
        return False
    block = blocks[0][1]

    block[0,3] = block[0,3] + BLOCK_X_OFFSET
    block[1,3] = block[1,3] + BLOCK_Y_OFFSET

    #sets the block height since 
    block[2,3] = 0.225 + BLOCK_Z_OFFSET

    above_block = deepcopy(block)
    above_block[2,3] = above_block[2,3] + ABOVE_DIST

    above_tower = deepcopy(tower_location)
    above_tower[2,3] = above_tower[2,3] + ABOVE_DIST

    #execute movements
    arm.exec_gripper_cmd(GRIPPER_OPEN_WIDTH)

    q_above_block, _, success, _ = ik.inverse(deepcopy(above_block), deepcopy(start_neutral), method='J_pseudo', alpha=.5)
    if not success:
        return False
    arm.safe_move_to_position(q_above_block)

    q_on_block, _, success, _ = ik.inverse(deepcopy(block), deepcopy(start_neutral), method='J_pseudo', alpha=.5)
    if not success:
        return False
    arm.safe_move_to_position(q_on_block)

    #close gripper
    arm.exec_gripper_cmd(GRIPPER_CLOSED_WIDTH)
    time.sleep(1)

    trajectory = True
    if trajectory:
        trajectory_planner = finalTrajectoryPlanner(fk, ik, arm)
        trajectory_planner.set_trajectory(lambda t: finalTrajectoryPlanner.circular_trajectory(t, block[0:3,3],goal_pos))
        trajectory_planner.follow_trajectory(q_on_block, 5, goal_neutral)
    else:
        arm.safe_move_to_position(start_neutral)
        arm.safe_move_to_position(goal_neutral)

    #moving onto tower
    q_on_tower, _, success, _ = ik.inverse(deepcopy(tower_location), deepcopy(goal_neutral), method='J_pseudo', alpha=.5)
    if not success:
        return False
    arm.safe_move_to_position(q_on_tower)

    #move off of tower
    arm.exec_gripper_cmd(GRIPPER_OPEN_WIDTH)
    arm.safe_move_to_position(goal_neutral)

    return True

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()

    #########REMOVE FOR LAB*****************
   # arm.open_gripper()
    #***************************
    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    #precomputed neutral positions
    if team == 'blue':
        BLOCK_X_OFFSET = 0#-0.01
        BLOCK_Y_OFFSET = 0#0.01
        BLOCK_Z_OFFSET = 0#-0.025
        start_y = 0.159
        goal_y = -0.159
        goal_pos = np.array([0.562, -0.159, 0.5])
        start_neutral = np.array([0.17869, 0.08683, 0.10313, -1.51866, -0.00893, 1.60503, 1.06713])
        goal_neutral = np.array([-0.13321, 0.08734, -0.1515, -1.51865, 0.01317, 1.60499, 0.50081])
        dynamic_neutral = np.array([-1.47733, 0.73124, -0.25751, -0.5396, 0.17965, 1.25898, 0.74095])
        q_start_middle = np.array([0.25182914, 0.2253079, 0.02475932, -2.05512443, -0.00729041, 2.28035246, 1.06611082])
        H_BASE_WORLD = np.array([[1,0,0,0],
                        [0,1,0,0.99],
                        [0,0,1,0],
                        [0,0,0,1]])
        H_WORLD_BASE = np.array([[1,0,0,0],
                                [0,1,0,-0.99],
                                [0,0,1,0],
                                [0,0,0,1]])
    else:
        BLOCK_X_OFFSET = 0
        BLOCK_Y_OFFSET = 0#0.01
        BLOCK_Z_OFFSET = 0
        start_y = -0.159
        goal_y = 0.159
        goal_pos = np.array([0.562, 0.159, 0.5])
        start_neutral = np.array([-0.13321, 0.08734, -0.1515, -1.51865, 0.01317, 1.60499, 0.50081])
        goal_neutral = np.array([0.17869, 0.08683, 0.10313, -1.51866, -0.00893, 1.60503, 1.06713])
        dynamic_neutral = np.array([1.5101, 0.72477, 0.16524, -0.54333, -0.11467, 1.2632, 0.81405])
        q_start_middle = np.array([-0.10451352, 0.22873289, -0.17755395, -2.05486711, 0.0527715, 2.27943692, 0.47350005])
        H_BASE_WORLD = np.array([[1,0,0,0],
                        [0,1,0,-0.99],
                        [0,0,1,0],
                        [0,0,0,1]])
        H_WORLD_BASE = np.array([[1,0,0,0],
                                [0,1,0,0.99],
                                [0,0,1,0],
                                [0,0,0,1]])


    tower_location = np.array([[[1,0,0,0.637],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.225],
                                [0,0,0,1]],
                                [[1,0,0,0.637],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.275],
                                [0,0,0,1]],
                                [[1,0,0,0.487],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.225],
                                [0,0,0,1]],
                                [[1,0,0,0.487],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.275],
                                [0,0,0,1]],
                                ])
    for i in range(4):
       look_for_static_blocks_and_stack(start_neutral, tower_location[i], goal_pos)

    print("Static blocks done. Moving onto dynamic blocks")


    tower_location = np.array([[[1,0,0,0.637],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.325],
                                [0,0,0,1]],

                                [[1,0,0,0.487],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.325],
                                [0,0,0,1]],

                                [[1,0,0,0.637],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.375],
                                [0,0,0,1]],

                                [[1,0,0,0.487],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.375],
                                [0,0,0,1]],

                                [[1,0,0,0.637],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.425],
                                [0,0,0,1]],

                                [[1,0,0,0.487],
                                [0,-1,0,goal_y],
                                [0,0,-1,0.425],
                                [0,0,0,1]],
                                ])
    count = 0
    for i in range(20):
        if count == 6:
            break
        if pick_up_dynamic_block(dynamic_neutral, detector, team):
            arm.safe_move_to_position(start_neutral)
            arm.safe_move_to_position(q_start_middle)
            arm.exec_gripper_cmd(GRIPPER_OPEN_WIDTH)
            look_for_static_blocks_and_stack(start_neutral, tower_location[count], goal_pos)
            count += 1

    # END STUDENT CODE