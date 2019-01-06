from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
# import cv2
import time

class TLClassifier(object):
    def __init__(self):
        
        self.graph = self.load_graph(rospy.get_param('~model'))
        self.input_operation = self.graph.get_operation_by_name('import/input')
        self.output_operation = self.graph.get_operation_by_name('import/final_result')
        self.labels = ['green', 'none', 'red', 'yellow']
        self.light_states = [TrafficLight.GREEN, TrafficLight.UNKNOWN, TrafficLight.RED, TrafficLight.YELLOW]
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=True))

    def load_graph(self, model_file):
        with tf.device('/gpu:0'):
            graph = tf.Graph()
            graph_def = tf.GraphDef()

            with open(model_file, "rb") as f:
                graph_def.ParseFromString(f.read())
            with graph.as_default():
                tf.import_graph_def(graph_def)

        return graph


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        tt = time.time()

        with tf.device('/gpu:0'):
            results = self.sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: image})

        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        label_idx = top_k[0]
        #rospy.logwarn("TLClassifier::get_classification: Label: {0}, TotalTime: {1}. Result: {2}".format(self.labels[label_idx], time.time()-tt, ['{0}:{1}'.format(self.labels[i],results[i]) for i in top_k]))
        return self.light_states[label_idx]
