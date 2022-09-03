#include "sign_detection.h"
#define SIGN_OFFSET_Y 0.09
#define SIGN_OFFSET_Z 0.013
#define SIGN_IMAGE_SIZE 0.025

// Esquina superior izquierda de la señal desde el centro de la etiqueta AR
tf::Vector3 P_ar_sign_topleft(
  -SIGN_IMAGE_SIZE,
  SIGN_OFFSET_Y+SIGN_IMAGE_SIZE,
  -SIGN_OFFSET_Z
);

// Esquina inferior derecha de la señal desde el centro de la etiqueta AR
tf::Vector3 P_ar_sign_bottomright(
  SIGN_IMAGE_SIZE,
  SIGN_OFFSET_Y-SIGN_IMAGE_SIZE,
  -SIGN_OFFSET_Z
);

HOGDescriptor hog(
    Size(64,64), //winSize
    Size(8,8), //blocksize
    Size(8,8), //blockStride,
    Size(8,8), //cellSize,
    9, //nbins,
    1, //derivAper,
    -1, //winSigma,
    0, //histogramNormType,
    0.2, //L2HysThresh,
    1,//gammal correction,
    64,//nlevels=64
    1);//Use signed gradients

int SZ = 64;

float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

void saveCameraInfo(const CameraInfoConstPtr& msg) {
    fx = msg->P[0];
    cx = msg->P[2];
    fy = msg->P[5];
    cy = msg->P[6];
    ROS_INFO("Camera parameters saved:\n fx = %f, fy = %f, cx = %f, cy = %f",fx,fy,cx,cy);
}

void saveMarkerPose(AlvarMarker &a) {
  traslation_c_ar = tf::Vector3(
    a.pose.pose.position.x,
    a.pose.pose.position.y,
    a.pose.pose.position.z);

  rotation_c_ar = tf::Quaternion(
    a.pose.pose.orientation.x,
    a.pose.pose.orientation.y,
    a.pose.pose.orientation.z,
    a.pose.pose.orientation.w);
}

float distance3D(tf::Vector3 v) {
  return sqrt((traslation_c_ar[0] * traslation_c_ar[0]) +
              (traslation_c_ar[1] * traslation_c_ar[1]) +
              (traslation_c_ar[2] * traslation_c_ar[2]));
}

void spaceToImage(tf::Vector3 &p, image_position_t &i) {
  if (p[2] > 0.001) {
    i.u = (fx*p[0])/p[2] + cx;
    i.v = (fy*p[1])/p[2] + cy;
  }
}

Mat deskew(Mat &img){
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2){
        return img.clone();
    }
    float skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

    return imgOut;
}

// Comprueba que la posición de la imagen calculada a partir del punto 3D se encuentra dentro del tamaño de la imagen
bool insideImage(image_position_t &p, cv_bridge::CvImagePtr &c) {
  bool inside = false;
  if(p.u > 0 && p.u < c->image.cols) {
    if(p.v > 0 && p.v < c->image.rows) {
      inside = true;
    }
  }
  return inside;
}

void packImgMsg(cv_bridge::CvImagePtr &cv_ptr, cv::Mat &image, std::string frame) {
  cv_ptr->header.stamp = ros::Time::now();
  cv_ptr->header.frame_id = frame;
  cv_ptr->encoding = "mono8";
  cv_ptr->image = image;
}

void classifySign() {
  resize(input_cv_ptr->image(cv::Rect(cv::Point(sign_topleft_image.u,sign_topleft_image.v),cv::Point(sign_bottomright_image.u,sign_bottomright_image.v))),inputCell,Size(SZ,SZ));
  cvtColor(inputCell,inputCell,COLOR_BGR2GRAY);
  packImgMsg(output_cv_ptr,inputCell,"cropped");
  cropped_pub.publish(output_cv_ptr->toImageMsg());
  
  Mat deskewedCell = deskew(inputCell);
  
  std::vector<float> testHOG;
  hog.compute(deskewedCell,testHOG);
  int descriptor_size = testHOG.size();
  Mat testMat(1,descriptor_size,CV_32FC1);  
  for(int j = 0;j<descriptor_size;j++){
    testMat.at<float>(0,j) = testHOG[j];
  }

  int sign = model->predict(testMat);
  double distance = sqrt((ar.pose.pose.position.x*ar.pose.pose.position.x)+
						 (ar.pose.pose.position.y*ar.pose.pose.position.y)+
						 (ar.pose.pose.position.z*ar.pose.pose.position.z));
  ROS_INFO("Distancia a la senal :%f metros",distance);
  switch(sign){
        case 0: 
        sign_msg.data = "Fin de prohibiciones"; 
        ROS_INFO( "Fin de prohibiciones");break;
        case 1: 
        sign_msg.data = "Entrada prohibida"; 
        ROS_INFO( "Entrada prohibida");break;
        case 2: 
        sign_msg.data = "Prohibido girar a la izquierda"; 
        ROS_INFO( "Prohibido girar a la izquierda");break;
        case 3: 
        sign_msg.data = "Prohibido girar a la derecha"; 
        ROS_INFO( "Prohibido girar a la derecha");break;
        case 4: 
        sign_msg.data = "Prohibido adelantar"; 
        ROS_INFO( "Prohibido adelantar");break; 
        case 5: 
        sign_msg.data = "Parking"; 
        ROS_INFO( "Parking");break;
        case 6: 
        sign_msg.data = "Paso de peatones"; 
        ROS_INFO( "Paso de peatones");break;
        case 7: 
        sign_msg.data = "Comienzo de obras"; 
        ROS_INFO( "Comienzo de obras");break;
        case 8: 
        sign_msg.data = "Stop"; 
        ROS_INFO( "Stop");break;
        
    } 
  sign_pub.publish(sign_msg);
  std::string str;
  str = sign_msg.data;
  cv::putText(input_cv_ptr->image,str,cv::Point(70,70),cv::FONT_HERSHEY_DUPLEX,2.5,cv::Scalar(255,0,0),4,false);
}

void drawDetection() {
  
  // Dibuja un círculo y un rectangulo en la etiqueta AR y el icono de la señal 
  cv::circle( input_cv_ptr->image, cv::Point(ar_image.u, ar_image.v), 10, CV_RGB( 255, 0, 0 ),2);
  cv::rectangle(input_cv_ptr->image, cv::Point(sign_topleft_image.u,sign_topleft_image.v),cv::Point(sign_bottomright_image.u,sign_bottomright_image.v),CV_RGB(255,0,255),2);
}

void callback(const ImageConstPtr &image_msg, const AlvarMarkersConstPtr &ar_msg) {
    try {
      input_cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
      for (int i = 0; i < ar_msg->markers.size(); ++i) {
        ar = ar_msg->markers[i];
        if (ar.id == 0) {
          saveMarkerPose(ar);
       
          if (distance3D(traslation_c_ar) <= 1.1) {
            T_c_ar = tf::Transform(rotation_c_ar, traslation_c_ar); //TF de la etiqueta AR a la cámara
              
            sign_topleft_world = T_c_ar * P_ar_sign_topleft;
            sign_bottomright_world = T_c_ar * P_ar_sign_bottomright;

            // Dada la calibración de la cámara, transforma los puntos 3D en píxeles.
            spaceToImage(traslation_c_ar, ar_image);              // Centro del AR
            spaceToImage(sign_topleft_world, sign_topleft_image); // dos esquinas de la señal para recortar
            spaceToImage(sign_bottomright_world, sign_bottomright_image);

            // Verifica que los puntos estén dentro de la imagen antes de recortar
            if (insideImage(sign_topleft_image, input_cv_ptr) && insideImage(sign_bottomright_image, input_cv_ptr)) {
              classifySign();
              drawDetection();
            }
          }
          else {
            ROS_INFO("Fuera de rango");
          }
        }
      }
      
    }
    catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  tracking_pub.publish(input_cv_ptr->toImageMsg());
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "sign_detection");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  
  nh.param<std::string>("/ar_topic",ar_topic,"/umagarage/ar_pose_marker");
  nh.param<std::string>("/image_topic",image_topic,"/adc_car/sensors/basler_camera/image_raw");
  nh.param<std::string>("/info_topic",info_topic,"/adc_car/sensors/basler_camera/camera_info");
  nh.param<std::string>("/sign_tracking_topic",tracking_topic,"/umagarage/sign_tracking");
  nh.param<std::string>("/sign_cropped_topic",cropped_topic,"/umagarage/sign_cropped");
  nh.param<std::string>("/sign_detection_topic",sign_topic,"/umagarage/sign_detection");
  
  nh.param<std::string>("/svm_path",svm_path,"/home/adrian/catkin_ws/src/umagarage/umagarage_tsd_v2/umagarage_tsd/models/linear_test2.yml");
  
  model = Algorithm::load<SVM>(svm_path);
  ROS_INFO("Kernel type     : %d",model->getKernelType());
  ROS_INFO("C               : %f",model->getC());
  ROS_INFO("Degree          : %f",model->getDegree());
  ROS_INFO("Nu              : %f",model->getNu());
  ROS_INFO("Gamma           : %f",model->getGamma());
  
  tracking_pub = it.advertise(tracking_topic,1);
  cropped_pub = it.advertise(cropped_topic,1);
  sign_pub = nh.advertise<std_msgs::String>(sign_topic,1);
  
  /* Sincroniza los mensajes provenientes de la cámara: image_raw, camera_info and AR detections */
  message_filters::Subscriber<Image> image_sub(nh, image_topic, 1);
  message_filters::Subscriber<AlvarMarkers> ar_sub(nh, ar_topic, 1);
  // ApproximateTime toma un tamaño de cola como argumento de su constructor MySyncPolicy(10)
  Synchronizer<SyncPolicy> sync(SyncPolicy(100), image_sub, ar_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  
  CameraInfoConstPtr info_msg = ros::topic::waitForMessage<CameraInfo>(info_topic);
  saveCameraInfo(info_msg);

  ros::spin();
  
  return 0;
}
