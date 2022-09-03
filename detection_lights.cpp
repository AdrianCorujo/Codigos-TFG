/*
  Software License Agreement (BSD License)

  Copyright (c) 2021, UMA GARAGE Autonomous Driving Team
  All rights reserved.

*/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

#include<sys/time.h>

#define MAX_CANDIDATES_CONSIDERED 5

#define CANDIDATE_PADDING 10
#define MS_TIMEOUT 700
#define MIN_RATIO 0.8
#define MAX_RATIO 1.2
#define MIN_AREA 100
#define MAX_AREA 10000

class preprocessor
{
	private:
		ros::NodeHandle nh;
		ros::NodeHandle n_;
		image_transport::ImageTransport it;
		image_transport::Subscriber imageSub;
		image_transport::Publisher imagePub;
		ros::Subscriber arSub;
		ros::Subscriber odomSub;
		ros::Publisher cmdPub;
		ros::Subscriber cmdNavSub;
		

	public:

		std::string image_sub_topic;
		std::string image_pub_topic;
		std::string ar_topic;
		std::string odom_topic;
		std::string cmd_topic;
		std::string cmd_nav_topic;
		

		double red_h_low_1, red_s_low_1, red_v_low_1;
		double red_h_high_1, red_s_high_1, red_v_high_1;
		double red_h_low_2, red_s_low_2, red_v_low_2;
		double red_h_high_2, red_s_high_2, red_v_high_2;
		double green_h_low, green_s_low, green_v_low;
		double green_h_high, green_s_high, green_v_high;
		double orange_h_low, orange_s_low, orange_v_low;
		double orange_h_high, orange_s_high, orange_v_high;

		struct position_t {
			double x;
			double y;
			double z;
		};

		position_t ar_position;
		std::vector<ar_track_alvar_msgs::AlvarMarker> markers;
		geometry_msgs::Twist cmd_msg;
		
		int ar_id;
		float cmd_nav_x;
		
		preprocessor():it(nh),n_(nh){
			
		    std::string node_name = ros::this_node::getName();
			ROS_INFO("Node name: %s",node_name.c_str());
			
			// Parametros de los topics
			n_.param<std::string>(node_name+"/image_sub_topic", image_sub_topic, "/adc_car/sensors/basler_camera/image_raw");
			n_.param<std::string>(node_name+"/image_pub_topic", image_pub_topic, "/umagarage/sign_tracking");
			n_.param<std::string>(node_name+"/ar_topic", ar_topic, "/ar_pose_marker");
			
			

			//Parametros luz semaforo rojo
			n_.param<double>(node_name+"/red_h_low_1", red_h_low_1, 0);
			n_.param<double>(node_name+"/red_s_low_1", red_s_low_1, 111);
			n_.param<double>(node_name+"/red_v_low_1", red_v_low_1, 85);
			
			n_.param<double>(node_name+"/red_h_high_1", red_h_high_1, 10);
			n_.param<double>(node_name+"/red_s_high_1", red_s_high_1, 255);
			n_.param<double>(node_name+"/red_v_high_1", red_v_high_1, 255);
			
			n_.param<double>(node_name+"/red_h_low_2", red_h_low_2, 170);
			n_.param<double>(node_name+"/red_s_low_2", red_s_low_2, 111);
			n_.param<double>(node_name+"/red_v_low_2", red_v_low_2, 85);
		
			n_.param<double>(node_name+"/red_h_high_2", red_h_high_2, 179);
			n_.param<double>(node_name+"/red_s_high_2", red_s_high_2, 255);
			n_.param<double>(node_name+"/red_v_high_2", red_v_high_2, 255);

			//Parametros luz semaforo verde
			n_.param<double>(node_name+"/green_h_low", green_h_low, 20);
			n_.param<double>(node_name+"/green_s_low", green_s_low, 111);
			n_.param<double>(node_name+"/green_v_low", green_v_low, 85);
			
			n_.param<double>(node_name+"/green_h_high", green_h_high, 75);
			n_.param<double>(node_name+"/green_s_high", green_s_high, 255);
			n_.param<double>(node_name+"/green_v_high", green_v_high, 255);

			//Parametros luz semaforo naranja
			n_.param<double>(node_name+"/orange_h_low", orange_h_low, 10);
			n_.param<double>(node_name+"/orange_s_low", orange_s_low, 111);
			n_.param<double>(node_name+"/orange_v_low", orange_v_low, 85);
			
			n_.param<double>(node_name+"/orange_h_high", orange_h_high, 20);
			n_.param<double>(node_name+"/orange_s_high", orange_s_high, 255);
			n_.param<double>(node_name+"/orange_v_high", orange_v_high, 255);

			imageSub = it.subscribe(image_sub_topic, 1, &preprocessor::imageCallBack, this);
			imagePub = it.advertise(image_pub_topic, 1);
			arSub = n_.subscribe(ar_topic,1,&preprocessor::arTagCallBack, this);
			
		}
  
		void arTagCallBack(const ar_track_alvar_msgs::AlvarMarkersConstPtr &msg) {
			markers = msg->markers;			
			
			if (markers.size() != 0) {
				ar_position.x = markers[0].pose.pose.position.x;
				ar_position.y = markers[0].pose.pose.position.y;
				ar_position.z = markers[0].pose.pose.position.z;
				ar_id = markers[0].id;
			}
		}

		void cmdNavCallBack(const geometry_msgs::TwistConstPtr &msg) {
			cmd_nav_x = msg->linear.x;
		}

		void imageCallBack(const sensor_msgs::ImageConstPtr &msg)
		{
			// Definición del puntero de enlace entre OpenCV y ROS
			cv_bridge::CvImagePtr cv_ptr;

			//Excepción => Manejo de msg con codificación de formato incorrecto
			// Si la imagen es bgr8 o mono8 => Evitamos copiar datos para que no caiga el nodo
			try {
				cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
			}
			catch(cv_bridge::Exception &e) {
				ROS_ERROR("cv_bridge exception: %s",e.what());
				return;
			}
			if (ar_id == 1) {
				bool found_light_green=false;
				bool found_light_red=false;
				bool found_light_orange=false;
				
				// CONVERSIÓN DE RGB A HSV //
				cv::Mat hsv_image;
				cv::cvtColor(cv_ptr->image, hsv_image, cv::COLOR_BGR2HSV);

				//Funciones detectoras//
				redDetector(cv_ptr, hsv_image, found_light_red);
				greenDetector(cv_ptr, hsv_image, found_light_green);
				orangeDetector(cv_ptr, hsv_image, found_light_orange);
				

				if (found_light_green || found_light_orange || found_light_red) {
					double distance = sqrt((ar_position.x*ar_position.x)+
										   (ar_position.y*ar_position.y)+
										   (ar_position.z*ar_position.z));
					ROS_INFO("Distancia al semaforo :%f metros",distance);
					if (distance <= 0.8) {
						if (found_light_red) {
							ROS_INFO("STOP! SEMAFORO EN ROJO!");
							cv::putText(cv_ptr->image,"STOP! SEMAFORO EN ROJO!",cv::Point(60,60),cv::FONT_HERSHEY_DUPLEX,2,cv::Scalar(0,0,255),4,false);
						}
						if (found_light_orange) {
							ROS_INFO("PRECAUCION, SEMAFORO EN AMBAR");
							cv::putText(cv_ptr->image,"PRECAUCION, SEMAFORO EN AMBAR",cv::Point(60,60),cv::FONT_HERSHEY_DUPLEX,2,cv::Scalar(0,128,255),4,false);
						}
						if (found_light_green) {
							ROS_INFO("CONTINUE, SEMAFORO EN VERDE");
							cv::putText(cv_ptr->image,"CONTINUE, SEMAFORO EN VERDE",cv::Point(60,60),cv::FONT_HERSHEY_DUPLEX,2,cv::Scalar(0,255,0),4,false);
						}
					}
				}
				ar_id = 0;
			}

			imagePub.publish(cv_ptr->toImageMsg());

		}
		void redDetector(const cv_bridge::CvImagePtr &cv_ptr, cv::Mat &hsv_image, bool &found_light_red)
		{
			// TIEMPO DE CÓMPUTO //
			struct timeval tp;
			gettimeofday(&tp, NULL);
			long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

			// FILTRACIÓN DEL COLOR ROJO //
			// Obtención de la máscara para el color rojo

			cv::Mat lower_mask;
			cv::Mat upper_mask;

			cv::inRange(hsv_image, cv::Scalar(red_h_low_1, red_s_low_1, red_v_low_1), cv::Scalar(red_h_high_1, red_s_high_1, red_v_high_1), lower_mask);
			cv::inRange(hsv_image, cv::Scalar(red_h_low_2, red_s_low_2, red_v_low_2),cv::Scalar(red_h_high_2, red_s_high_2, red_v_high_2), upper_mask);

			cv::Mat combination;
			cv::addWeighted(lower_mask, 1.0, upper_mask, 1.0, 0.0, combination);
						
			// BÚSQUEDA DE CONTORNOS ROJOS //
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(
					combination,
					contours,
					CV_RETR_EXTERNAL, 
					CV_CHAIN_APPROX_SIMPLE 
					);

			// ELMINACIÓN DE FALSOS POSITIVOS Y NEGATIVOS //
			int candidates_considered = 0;
			for(size_t i = 0 ; i < contours.size(); i++) 
			{
				int area = cv::contourArea(contours[i]);

				if(area >= MIN_AREA && area <= MAX_AREA) {
					if(candidates_considered >= MAX_CANDIDATES_CONSIDERED) {
						break;
					} else {
						candidates_considered++;
					}
					// Definición del rectángulo de visualización
					cv::Rect rect = cv::boundingRect(contours[i]);

					// LOCALIZACIÓN DE LA LUZ DEL SEMÁFORO //
					// Cada candidato será una subimagen de la imagen original.
					int sub_x = std::max(rect.x-CANDIDATE_PADDING, 0);
					int sub_y = std::max(rect.y-CANDIDATE_PADDING, 0);
					int sub_width = std::min(rect.width+ 2*CANDIDATE_PADDING, combination.cols - sub_x);
					int sub_height = std::min(rect.height+ 2*CANDIDATE_PADDING, combination.rows - sub_y);
					cv::Mat subimage = cv::Mat(cv_ptr->image, cv::Rect(sub_x, sub_y, sub_width, sub_height));

					cv::rectangle(cv_ptr->image, cv::Rect(sub_x, sub_y, sub_width, sub_height), cv::Scalar(0), 2, 8, 0);

					// img es la principal imagen => Con la que trabajamos
					dlib::array2d<dlib::rgb_pixel> img;
					dlib::assign_image(img,dlib::cv_image<dlib::bgr_pixel>(subimage));

					// La imagen candidata la desmuestreo para asegurar encontrar todos los posibles semáforos
					dlib::pyramid_up(img);
					dlib::pyramid_up(img);

					long int height;
					long int width;
					height = img.nc();
					width = img.nr();

					// Uso un rectangulo delimitador simplificado para ajustar al máximo
					cv::rectangle(cv_ptr->image, rect, cv::Scalar(255), 3, 8, 0);
					found_light_red = true;
				}

				// Captación de tiempo
				gettimeofday(&tp, NULL);
				long int ms_now = tp.tv_sec * 1000 + tp.tv_usec / 1000;
				if(ms_now - ms >= MS_TIMEOUT) {
					ROS_INFO("Timeout red.");
					break;
				}
				
			}

		}
		void greenDetector(const cv_bridge::CvImagePtr &cv_ptr, cv::Mat &hsv_image, bool &found_light_green)
		{			
			// TIEMPO DE CÓMPUTO //
			struct timeval tp;
			gettimeofday(&tp, NULL);
			long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
			
			// FILTRACIÓN DEL COLOR VERDE //
			// Obtención de la máscara para el color verde

			cv::Mat combination;
			cv::inRange(hsv_image, cv::Scalar(green_h_low, green_s_low, green_v_low), cv::Scalar(green_h_high, green_s_high, green_v_high), combination);

			// BÚSQUEDA DE CONTORNOS VERDES //
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(
					combination,
					contours,
					CV_RETR_EXTERNAL, 
					CV_CHAIN_APPROX_SIMPLE
					);

			// ELMINACIÓN DE FALSOS POSITIVOS Y NEGATIVOS //
			int candidates_considered = 0;
			for(size_t i = 0 ; i < contours.size(); i++) 
			{
				int area = cv::contourArea(contours[i]);

				if(area >= MIN_AREA && area <= MAX_AREA) {
					if(candidates_considered >= MAX_CANDIDATES_CONSIDERED) {
						break;
					} else {
						candidates_considered++;
					}
					// Definición del rectángulo de visualización
					cv::Rect rect = cv::boundingRect(contours[i]);

					// LOCALIZACIÓN DE LA LUZ DEL SEMÁFORO //
					//  Cada candidato será una subimagen de la imagen original.
					int sub_x = std::max(rect.x-CANDIDATE_PADDING, 0);
					int sub_y = std::max(rect.y-CANDIDATE_PADDING, 0);
					int sub_width = std::min(rect.width+ 2*CANDIDATE_PADDING, combination.cols - sub_x);
					int sub_height = std::min(rect.height+ 2*CANDIDATE_PADDING, combination.rows - sub_y);
					cv::Mat subimage = cv::Mat(cv_ptr->image, cv::Rect(sub_x, sub_y, sub_width, sub_height));

					cv::rectangle(cv_ptr->image, cv::Rect(sub_x, sub_y, sub_width, sub_height), cv::Scalar(0), 2, 8, 0);

					// img es la principal imagen => Con la que trabajamos
					dlib::array2d<dlib::rgb_pixel> img;
					assign_image(img,dlib::cv_image<dlib::bgr_pixel>(subimage));

					// La imagen candidata la desmuestreo para asegurar encontrar todos los posibles semáforos
					pyramid_up(img);
					pyramid_up(img);

					long int height;
					long int width;
					height = img.nc();
					width = img.nr();

					// Uso un rectangulo delimitador simplificado para ajustar al máximo
					cv::rectangle(cv_ptr->image, rect, cv::Scalar(255), 3, 8, 0);
					found_light_green = true;
				}

				gettimeofday(&tp, NULL);
				long int ms_now = tp.tv_sec * 1000 + tp.tv_usec / 1000;
				if(ms_now - ms >= MS_TIMEOUT) {
					ROS_INFO("Timeout green.");
					break;
				}
					
			}

		}
		
		void orangeDetector(const cv_bridge::CvImagePtr &cv_ptr, cv::Mat &hsv_image, bool &found_light_orange)
		{
	
			// TIEMPO DE CÓMPUTO //
			struct timeval tp;
			gettimeofday(&tp, NULL);
			long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
			
			// FILTRACIÓN DEL COLOR NARANJA //
			// Obtención de la máscara para el color naranja

			cv::Mat combination;

			cv::inRange(hsv_image, cv::Scalar(orange_h_low, orange_s_low, orange_v_low), cv::Scalar(orange_h_high, orange_s_high, orange_v_high), combination);
			// BÚSQUEDA DE CONTORNOS NARANJAS //
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(
					combination,
					contours,
					CV_RETR_EXTERNAL, // Only use outer contours
					CV_CHAIN_APPROX_SIMPLE // Compress heavily overlapping contours
					);
			// ELMINACIÓN DE FALSOS POSITIVOS Y NEGATIVOS //
			int candidates_considered = 0;
			for(size_t i = 0 ; i < contours.size(); i++) 
			{
				int area = cv::contourArea(contours[i]);

				if(area >= MIN_AREA && area <= MAX_AREA) {
					if(candidates_considered >= MAX_CANDIDATES_CONSIDERED) {
						break;
					} else {
						candidates_considered++;
					}
					// Definición del rectángulo de visualización
					cv::Rect rect = cv::boundingRect(contours[i]);

					// LOCALIZACIÓN DE LA LUZ DEL SEMÁFORO //
					//  Cada candidato será una subimagen de la imagen original.
					int sub_x = std::max(rect.x-CANDIDATE_PADDING, 0);
					int sub_y = std::max(rect.y-CANDIDATE_PADDING, 0);
					int sub_width = std::min(rect.width+ 2*CANDIDATE_PADDING, combination.cols - sub_x);
					int sub_height = std::min(rect.height+ 2*CANDIDATE_PADDING, combination.rows - sub_y);
					cv::Mat subimage = cv::Mat(cv_ptr->image, cv::Rect(sub_x, sub_y, sub_width, sub_height));

					cv::rectangle(cv_ptr->image, cv::Rect(sub_x, sub_y, sub_width, sub_height), cv::Scalar(0), 2, 8, 0);

					// img es la principal imagen => Con la que trabajamos
					dlib::array2d<dlib::rgb_pixel> img;
					dlib::assign_image(img,dlib::cv_image<dlib::bgr_pixel>(subimage));

					// La imagen candidata la desmuestreo para asegurar encontrar todos los posibles semáforos
					pyramid_up(img);
					pyramid_up(img);

					long int height;
					long int width;
					height = img.nc();
					width = img.nr();

					// Uso un rectangulo delimitador simplificado para ajustar al máximo
					cv::rectangle(cv_ptr->image, rect, cv::Scalar(255), 3, 8, 0);
					found_light_orange = true;
				}

				gettimeofday(&tp, NULL);
				long int ms_now = tp.tv_sec * 1000 + tp.tv_usec / 1000;
				if(ms_now - ms >= MS_TIMEOUT) {
					ROS_INFO("Timeout orange.");
					break;
				}
					
			}

		}
	
	
};

int main(int argc, char** argv)
{
	ros::init(argc,argv, "detection_lights");
	preprocessor p;
	ros::spin();
	return 0;
}
