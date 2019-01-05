from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import cv2
import time

class TLClassifier(object):
    def __init__(self):
        
        self.graph = self.load_graph(rospy.get_param('~model'))
        self.input_operation = self.graph.get_operation_by_name('import/input')
        self.output_operation = self.graph.get_operation_by_name('import/final_result')
        self.labels = ['green', 'none', 'red', 'yellow']


    def load_graph(self, model_file):
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
        with tf.Session(graph=self.graph) as sess:
            t1 = time.time()
            input_height = 224
            input_width = 224
            input_mean = 0
            input_std = 255

            #float_caster = tf.cast(image_reader, tf.float32)
            image = cv2.resize(image, dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)
            np_image = np.asarray(image)
            dims_expander = tf.expand_dims(np_image, 0)
            resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
            normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

            t2 = time.time()
            tensor = sess.run(normalized)
            t3 = time.time()
            results = sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: tensor})
            
            t4 = time.time()
            results = np.squeeze(results)
            t5 = time.time()
            top_k = results.argsort()[-5:][::-1]

            rospy.logwarn("TLClassifier::get_classification: Result: {0}".format(['{0}:{1}'.format(self.labels[i],results[i]) for i in top_k]))
            rospy.logwarn("TLClassifier::get_classification: Time {0} {1} {2} {3} {4} {5} T:{6}".format(t1-tt, t2-t1, t3-t2, t4-t3, t5-t4, time.time()-t5, time.time()-tt))
            final_result = self.labels[top_k[0]]
            if final_result == "green":
                rospy.loginfo('TLClassifier::get_classification - Got a GREEN')
                return TrafficLight.GREEN
            elif final_result == "red":
                rospy.loginfo('TLClassifier::get_classification - Got a RED')
                return TrafficLight.RED
            elif final_result == "yellow":
                rospy.loginfo('TLClassifier::get_classification - Got a YELLOW')
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
