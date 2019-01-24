#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import time

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
LOOP_RATE = 20 # Processing Frequency.
MAX_DECEL = 5.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Member variables
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.current_pose = None
        self.current_wp = -1
        self.stopline_wp_idx = -1
        self.closest_waypoint_idx = -1
        self.traffic_light_updated = False
        self.linear_velocity = 0

        # Debug vars
        self.current_pose_t = time.time()
        self.loop_t = time.time()
        
        self.loop()

    def loop(self):
        rate = rospy.Rate(LOOP_RATE)
        while not rospy.is_shutdown():
            if self.base_waypoints:
                if self.current_wp != self.closest_waypoint_idx or self.traffic_light_updated:
                    self.closest_waypoint_idx = self.current_wp
                    #newtime = time.time()
                    #rospy.logwarn('WaypointUpdater::loop - Time: {0} Closest: {1} Velocity:{2}'.format(newtime-self.loop_t, self.closest_waypoint_idx, self.linear_velocity))
                    #self.loop_t = newtime
                    self.publish_waypoints()

                self.traffic_light_updated = False
            rate.sleep()


    def get_closest_waypoint_idx(self):
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            # closest_idx was behind the vehicle. Choose next
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx


    def pose_cb(self, msg):
        self.current_pose = msg
        self.current_wp = self.get_closest_waypoint_idx()
        #newtime = time.time()
        #rospy.logwarn('WaypointUpdater::pose_cb Time since last pose: {0}'.format(newtime-self.current_pose_t))
        #self.current_pose_t = newtime


    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

            rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
            rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
            #rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)


    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message.
        self.stopline_wp_idx = msg.data
        self.traffic_light_updated = True
        #rospy.logwarn('WaypointUpdater::traffic_cb {0}'.format(self.stopline_wp_idx))


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass


    def current_velocity_cb(self, msg):
        self.linear_velocity = msg.twist.linear.x


    def publish_waypoints(self):
        lane = Lane()
        lane.header = self.base_waypoints.header

        farthest_idx = self.closest_waypoint_idx + LOOKAHEAD_WPS
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = self.base_waypoints.waypoints[self.closest_waypoint_idx:farthest_idx]
        else:
            lane_wp = self.base_waypoints.waypoints[self.closest_waypoint_idx:farthest_idx]
            lane.waypoints = self.decelerate_waypoints(lane_wp)

        self.final_waypoints_pub.publish(lane)


    def decelerate_waypoints(self, waypoints):
        #rospy.logwarn('WaypointUpdater::decelerate_waypoints BEGIN {0}'.format(len(waypoints)))
        # Choose where to stop
        stop_idx = max(self.stopline_wp_idx - self.closest_waypoint_idx - 6, 0)
        stop_wp = waypoints[stop_idx]
        #rospy.logwarn('WaypointUpdater::decelerate_waypoints From {0} To {1} StopAt {2}-{3}'.format(self.closest_waypoint_idx, self.closest_waypoint_idx+len(waypoints)-1, self.stopline_wp_idx, stop_idx))
        decel_wp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            #rospy.logwarn('WaypointUpdater::decelerate_waypoints1 ({0}[{1}]: V[{2}]-WpV[{3}]-Linear_vel[{4}]) '.format(i, self.closest_waypoint_idx+i, vel, wp.twist.twist.linear.x, p.twist.twist.linear.x))

            # Create the published waypoints list
            decel_wp.append(p)

        return decel_wp


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


    def distance2(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
