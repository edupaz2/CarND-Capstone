from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import cv2

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

    def read_tensor_from_image_file(file_name, input_height=128, input_width=128, input_mean=0, input_std=255):
        input_name = "file_reader"
        output_name = "normalized"
        file_reader = tf.read_file(file_name, input_name)
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with tf.Session(graph=self.graph) as sess:
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

            tensor = sess.run(normalized)
            results = sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: tensor})
    
            results = np.squeeze(results)
            top_k = results.argsort()[-5:][::-1]

            rospy.logwarn("TLClassifier::get_classification: {0}".format([results[i] for i in top_k]))
            final_result = self.labels[top_k[0]]
            if final_result == "green":
                return TrafficLight.GREEN
            elif final_result == "red":
                return TrafficLight.RED
            elif final_result == "yellow":
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
