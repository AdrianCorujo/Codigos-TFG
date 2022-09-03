#include <iostream>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/String.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>
#include <ar_track_alvar_msgs/AlvarMarker.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tf/tf.h>

using namespace sensor_msgs;
using namespace message_filters;
using namespace ar_track_alvar_msgs;
using namespace cv::ml;
using namespace cv;
using namespace std;

std::string image_topic, info_topic, ar_topic, tracking_topic, cropped_topic, svm_path, sign_topic;
image_transport::Publisher tracking_pub, cropped_pub;
ros::Publisher sign_pub;

typedef sync_policies::ApproximateTime<Image, AlvarMarkers> SyncPolicy;

// Sistema de coordenadas propio de la imagen
struct image_position_t{
  int u;
  int v;
};

cv_bridge::CvImagePtr input_cv_ptr;
cv_bridge::CvImagePtr output_cv_ptr(new cv_bridge::CvImage);
std_msgs::String sign_msg;

tf::Vector3 traslation_c_ar; // Traslacion del marco de la etiqueta AR al marco de la cámara
tf::Quaternion rotation_c_ar; // Rotación del marco de la etiqueta AR al marco de la cámara
tf::Transform T_c_ar; // Transforma la matriz homogénea del marco de la etiqueta AR al marco de la cámara

tf::Vector3 sign_topleft_world, sign_bottomright_world;
double dist,angle;
double fx, cx;
double fy, cy;
image_position_t ar_image, sign_topleft_image, sign_bottomright_image;

Ptr<SVM> model;

Mat inputCell, deskewedCell;
std::string sign_str;

AlvarMarker ar;
