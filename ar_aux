/*
  Software License Agreement (BSD License)

  Description: This node splits the detections made by alvar_ar_tracker, which are sent in a 
  single message in form of an array, into a sequence of messages. This is neccesary to syncronize the 
  detections to the corresponding frames in other nodes with message_filters package, because the array has no
  timestamp.
*/

#include <ros/ros.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>

std::string ar_topic_sub;
std::string ar_topic_pub;
ros::Subscriber arSub;
ros::Publisher arPub;
ar_track_alvar_msgs::AlvarMarkers markers;


void callback(const ar_track_alvar_msgs::AlvarMarkersConstPtr& msg) {
  markers.header = msg->header;
  markers.markers = msg->markers;
  if(msg->markers.size() >0) {  
    markers.header.stamp = msg->markers[0].header.stamp;
  }
  arPub.publish(markers);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "umagarage_ar_aux");
  ros::NodeHandle nh;

  nh.param<std::string>("/alvar_ar_topic",ar_topic_sub,"/ar_pose_marker");
  nh.param<std::string>("/umagarage_ar_topic",ar_topic_pub,"/umagarage/ar_pose_marker");
  
  arSub = nh.subscribe(ar_topic_sub,1,&callback);
  arPub = nh.advertise<ar_track_alvar_msgs::AlvarMarkers>(ar_topic_pub,1);
  
  ros::spin();
  return 0;
}
