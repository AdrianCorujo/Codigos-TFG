
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

string pathName = "/home/adrian/catkin_ws/src/umagarage/umagarage_tsd_v2/umagarage_tsd/datasets/tsd_v2_test2";
double training_perc = 0.7;

int SZ = 64;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;


Mat deskew(Mat& img){
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

void loadTrainTestLabel(string &pathName, int sign, int dataset_size, vector<Mat> &trainCells, vector<Mat> &testCells,vector<int> &trainLabels, vector<int> &testLabels)
{
    int training_n = round(training_perc*dataset_size);
    cout << training_n << endl;
    string sign_str;
    switch(sign){
        case 0: sign_str = "Fin de prohibiciones"; break;
        case 1: sign_str = "Entrada prohibida"; break;
        case 2: sign_str = "Prohibido girar a la izquierda"; break;
        case 3: sign_str = "Prohibido girar a la derecha"; break;
        case 4: sign_str = "Prohibido adelantar"; break;
        case 5: sign_str = "parking"; break;
        case 6: sign_str = "Paso de peatones"; break;
        case 7: sign_str = "Comienzo de obras"; break;
        case 8: sign_str = "stop"; break;
        
    }
    cout << sign_str << endl;
    
    for(int i = 0; i < dataset_size; i++) {
        Mat digitImg = imread(pathName+"/"+sign_str+"/"+to_string(i)+".jpg",0);
        Mat resized;
        resize(digitImg,resized,Size(64,64)); 
        if(i<training_n){
            trainCells.push_back(resized);
            trainLabels.push_back(sign);
        }
        else {
            testCells.push_back(resized);
            testLabels.push_back(sign);
        }
    }
    cout << trainLabels[0] << "\t" << trainLabels[2] << "\t" << trainLabels[training_n] << "\t" << trainLabels[trainLabels.size()-1] << endl;
    cout << "Entrenadas " << sign_str << " imagenes: " << trainLabels.size() << endl;
    cout << "Testeadas " << sign_str << " imagenes: " << testLabels.size() << endl;
}

void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells,vector<Mat> &deskewedTestCells, vector<Mat> &trainCells, vector<Mat> &testCells){

    for(int i=0;i<trainCells.size();i++){

        Mat deskewedImg = deskew(trainCells[i]);
        deskewedTrainCells.push_back(deskewedImg);
    }

    for(int i=0;i<testCells.size();i++){

        Mat deskewedImg = deskew(testCells[i]);
        deskewedTestCells.push_back(deskewedImg);
    }
}

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


void CreateTrainTestHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainCells, vector<Mat> &deskewedtestCells){

    for(int y=0;y<deskewedtrainCells.size();y++){
        vector<float> descriptors;
        hog.compute(deskewedtrainCells[y],descriptors);
        trainHOG.push_back(descriptors);
    }

    for(int y=0;y<deskewedtestCells.size();y++){

        vector<float> descriptors;
        hog.compute(deskewedtestCells[y],descriptors);
        testHOG.push_back(descriptors);
    }
}

void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{
    int descriptor_size = trainHOG[0].size();

    for(int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j];
        }
    }
    for(int i = 0;i<testHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j];
        }
    }
}

void getSVMParams(SVM *svm)
{
    cout << "Kernel type     : " << svm->getKernelType() << endl;
    cout << "Type            : " << svm->getType() << endl;
    cout << "C               : " << svm->getC() << endl;
    cout << "Degree          : " << svm->getDegree() << endl;
    cout << "Nu              : " << svm->getNu() << endl;
    cout << "Gamma           : " << svm->getGamma() << endl;
}

Ptr<SVM> svmInit(float C, float gamma)
{
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(SVM::LINEAR);
  svm->setType(SVM::C_SVC);

  return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  svm->train(td);
  svm->save("/home/adrian/catkin_ws/src/umagarage/umagarage_tsd_v2/umagarage_tsd/models/linear_test2.yml");
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat )
{
  svm->predict(testMat, testResponse);
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    if(testResponse.at<float>(i,0) == testLabels[i])
      count = count + 1;
  }
  accuracy = (count/testResponse.rows)*100;
}


int main()
{
    vector<Mat> trainCells;
    vector<Mat> testCells;
    vector<int> trainLabels;
    vector<int> testLabels;
    loadTrainTestLabel(pathName,0,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,1,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,2,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,3,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,4,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,5,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,6,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,7,54,trainCells,testCells,trainLabels,testLabels);
    loadTrainTestLabel(pathName,8,54,trainCells,testCells,trainLabels,testLabels);
    vector<Mat> deskewedTrainCells;
    vector<Mat> deskewedTestCells;
    CreateDeskewedTrainTest(deskewedTrainCells,deskewedTestCells,trainCells,testCells);
    
    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > testHOG;
    CreateTrainTestHOG(trainHOG,testHOG,deskewedTrainCells,deskewedTestCells);
    
    int descriptor_size = trainHOG[0].size();
    cout << "Tamaño del descriptor : " << descriptor_size << endl;

    Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
    Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);

    ConvertVectortoMatrix(trainHOG,testHOG,trainMat,testMat);
    float C = 20, gamma = 0.5;

    Mat testResponse;
    Ptr<SVM> model = svmInit(C, gamma);

    ///////////  SVM Training  ////////////////
    svmTrain(model, trainMat, trainLabels);
    ///////////  SVM Testing  ////////////////
    svmPredict(model, testResponse, testMat);
    cout << "Tamaño matriz dectesteo : "<< testMat.size() << endl;
    cout << "Tamaño matriz de respuesta : "<< testResponse.size() << endl;
    ////////////// Find Accuracy   ///////////
    float count = 0;
    float accuracy = 0 ;
    getSVMParams(model);
    SVMevaluate(testResponse, count, accuracy, testLabels);

    cout << "La precision es del :" << accuracy << endl;

    return 0;
}
