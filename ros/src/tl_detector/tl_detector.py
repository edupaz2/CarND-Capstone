#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
from cv2 import imwrite
import time

STATE_COUNT_THRESHOLD = 3
LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
LOOP_RATE = 50 # Processing Frequency. Same as waypoint_follower/pure_pursuit.cpp LOOP_RATE

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # Member variables
        self.testing = True

        self.current_pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.lights_closest_waypoint = []

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.light_wp = -1
        self.state = TrafficLight.UNKNOWN
        self.state_count = 0
        self.image_timestamp = time.time()

        self.loop()


    def loop(self):
        rate = rospy.Rate(LOOP_RATE)
        while not rospy.is_shutdown():
            rate.sleep()


    def pose_cb(self, msg):
        self.current_pose = msg


    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        if self.waypoint_tree and len(self.lights) == 0:
            self.lights = msg.lights
            self.preprocess_traffic_lights()


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        current_t = time.time()
        if current_t - self.image_timestamp < 1.0:
            return

        self.image_timestamp = current_t

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        #rospy.logwarn('TLDetector::image_cb. Lightwp: {0}-{1} State: {2}-{3}'.format(light_wp, self.light_wp, state, self.state))
        if self.state != state or light_wp != self.light_wp:
            self.state_count = 0
            self.state = state
            self.light_wp = light_wp
        elif self.state_count == STATE_COUNT_THRESHOLD:
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
            self.upcoming_red_light_pub.publish(Int32(light_wp))

        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx


    def get_light_state(self, light, wp):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # For testing, just return the light state
        #if(self.testing):
        #    return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        #rospy.logwarn('TLDetector::get_light_state - New image incoming')

        lightState = TrafficLight.UNKNOWN
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        lightState = self.light_classifier.get_classification(cv_image)
        #rospy.logwarn('TLDetector::get_light_state - Processing result: {0}'.format(lightState))

        # For dumping images to files for training the model
        #res = cv2.imwrite('/home/student/workspace/CarND-Capstone/imgs/sim/img_{0}_{1}.png'.format(wp, self.image_counter), cv_image)

        #Get classification
        return lightState

    def preprocess_traffic_lights(self):
        rospy.loginfo('TLDetector::preprocess_traffic_lights PREPROCESSING')

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        #find the closest visible traffic light (if one exists)
        for i, light in enumerate(self.lights):
            # Get stop line waypoint index
            line = stop_line_positions[i]
            temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
            self.lights_closest_waypoint.append(temp_wp_idx)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.waypoint_tree:
            return -1, TrafficLight.UNKNOWN

        closest_light = None
        line_wp_idx = None

        if(self.current_pose):
            car_wp_idx = self.get_closest_waypoint(self.current_pose.pose.position.x, self.current_pose.pose.position.y)

        #find the closest visible traffic light (if one exists)
        lane_wp = self.base_waypoints.waypoints[car_wp_idx:car_wp_idx + LOOKAHEAD_WPS]
        diff = len(lane_wp)
        for i, light in enumerate(self.lights):
            temp_wp_idx = self.lights_closest_waypoint[i]
            # Find closest stop line waypoint index
            d = temp_wp_idx - car_wp_idx
            if d >= 0 and d < diff:
                diff = d
                closest_light = light
                line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light, line_wp_idx)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
