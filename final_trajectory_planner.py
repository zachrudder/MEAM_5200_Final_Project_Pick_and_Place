import numpy as np
from core.utils import time_in_seconds
from lib.calcAngDiff import calcAngDiff
from lib.IK_velocity_null import IK_velocity_null
from lib.IK_position_null import IK


LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
CENTER = LOWER + (UPPER - LOWER) / 2

class finalTrajectoryPlanner():

    def __init__(self, fk, ik, arm):
        self.fk = fk
        self.ik = ik
        self.ik_pos = IK()
        self.arm = arm
        self.trajectory = None
    
    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    @staticmethod
    def circular_trajectory(t, start, goal):
        """creates a circular trajectory where the start 
        has a derivative that is completely veritcle"""
        unit_z = np.array([0,0,1])
        circle_plane = np.cross(unit_z, start - goal)
        circle_plane = circle_plane / np.linalg.norm(circle_plane)

        unit_u = np.cross(unit_z, circle_plane)

        unit_p0_p1 = goal-start
        unit_p0_p1 = unit_p0_p1 / np.linalg.norm(unit_p0_p1)

        u_proj_po_p1 = np.dot(unit_u, unit_p0_p1)

        radius = np.linalg.norm(goal-start) / (2*u_proj_po_p1)
        center = start + (unit_u * radius)

        theta = np.arccos(u_proj_po_p1)
        scale = np.pi - 2*theta

        x = center + radius*(np.cos(scale*t)*-unit_u + np.sin(scale*t)*unit_z)
        x_dot = radius*(np.sin(scale*t)*unit_u + np.cos(scale*t)*unit_z)

        Rdes = np.diag([1., -1., -1.])
        ang_vdes = np.array([0,0,0])

        return Rdes, ang_vdes, x, np.array([0,0,0])



    @staticmethod
    def create_cubic_bezier_trajectory(t, p0, p1, p2, p3):
        """takes 4 points and creates a cubic bezier trajectory
        
        Args:
            p0: starting point
            p1: first control point
            p2: second control point
            p3: ending point
            
        Returns:
            Rdes, ang_vdes, x, x_dot = f(t)(lambda function)"""
        if t > 1:
            return np.diag([1., -1., -1.]), np.array([0,0,0]), p3, np.array([0,0,0])
        x = ((1-t)**3)*p0 + 3*t*((1-t)**2)*p1 + 3*(t**2)*(1-t)*p2 + (t**3)*p3
        x_dot = (3 * (1 - t)**2 * (p1 - p0) + 
                6 * (1 - t) * t * (p2 - p1) + 
                3 * t**2 * (p3 - p2))
        
        Rdes = np.diag([1., -1., -1.])
        ang_vdes = np.array([0,0,0])
        
        return Rdes, ang_vdes, x, x_dot


    @staticmethod
    def line(t, p0, p1):
        Rdes = np.diag([1., -1., -1.])
        ang_vdes = np.array([0,0,0])

        x = (p1-p0)*t + p0
        x_dot = (p1-p0)

        return Rdes, ang_vdes, x, np.array([0,0,0])

    def follow_trajectory(self, q, total_time, goal_configuration):
        start_time = time_in_seconds()
        last_time = start_time
        while time_in_seconds() - start_time - 1 <= total_time:

            if np.linalg.norm(q-goal_configuration) < 0.1:
                print("Finished trajectory")
                break

            t = time_in_seconds() - start_time

            # get desired trajectory position and velocity
            Rdes, ang_vdes, xdes, vdes = self.trajectory(t/total_time)

            _, T0e = self.fk.forward(q)

            R = (T0e[:3,:3])
            x = (T0e[0:3,3])
            curr_x = np.copy(x.flatten())

            # First Order Integrator, Proportional Control with Feed Forward
            kp = 5.0
            v = vdes + kp * (xdes - curr_x)
            
            # Rotation
            kr = 1.0
            omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()

            ## STUDENT CODE MODIFY HERE, DEFINE SECONDARY TASK IN THE NULL SPACE

            # Velocity Inverse Kinematics
            k0 = 1.0
            b = - k0 * (q - CENTER)
            dq = IK_velocity_null(q,v, omega, b).flatten()


            # Get the correct timing to update with the robot
            dt = time_in_seconds() - last_time
            new_q = q + dt * dq
            self.arm.safe_set_joint_positions_velocities(new_q, dq)

            q = new_q
            last_time = time_in_seconds()

    def follow_trajectory_codys_version(self, q, total_time, goal_configuration):
        start_time = time_in_seconds()
        t = 0
        while time_in_seconds() - start_time - 1 <= total_time:

            if np.linalg.norm(q-goal_configuration) < 0.1:
                print("Finished trajectory")
                break

            t = time_in_seconds() - start_time

            # get desired trajectory position and velocity
            Rdes, ang_vdes, xdes, vdes = self.trajectory(t/total_time)

            target = np.vstack((np.hstack((Rdes, np.array(xdes).reshape(3,1))), np.array([0, 0, 0, 1])))
            new_q, _, success, _ = self.ik_pos.inverse(target, q, 'J_pseudo', 1)
            print("attempt")
            if success:
                print("sucess")
                dq = new_q - q
                q = new_q

            self.arm.safe_set_joint_positions_velocities(new_q, dq)

            q = new_q

