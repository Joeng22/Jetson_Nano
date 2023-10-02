#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//This header includes definition of 'rectangle()' function//
#include<opencv2/objdetect/objdetect.hpp>
//This header includes the definition of Cascade Classifier//
#include<string>
using namespace std;
using namespace cv;
int main(int argc, char** argv) {
   Mat video_stream;//Declaring a matrix hold frames from video stream//
   Mat ImageGray;
   VideoCapture real_time(0);//capturing video from default webcam//
   namedWindow("Face Detection");//Declaring an window to show the result//


   string trained_classifier_location = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";//Defining the location our XML Trained Classifier in a string//
   string trained_classifier_smile = "/usr/share/opencv4/haarcascades/haarcascade_smile.xml";


   CascadeClassifier faceDetector;//Declaring an object named 'face detector' of CascadeClassifier class//
   CascadeClassifier smileDetector;
   
   faceDetector.load(trained_classifier_location);//loading the XML trained classifier in the object//
   smileDetector.load(trained_classifier_smile);


   vector<Rect>faces;//Declaring a rectangular vector named faces//
   while (true) {

    real_time.read(video_stream);// reading frames from camera and loading them in 'video_stream' Matrix//

    cvtColor(video_stream,ImageGray,COLOR_BGR2GRAY);
    faceDetector.detectMultiScale(ImageGray, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(30, 30));//Detecting the faces in 'image_with_humanfaces' matrix//


      for (int i = 0; i < faces.size(); i++){ //for locating the face
        Mat faceROI = ImageGray(faces[i]);//Storing face in the matrix//
        int x = faces[i].x;//Getting the initial row value of face rectangle's starting point//
        int y = faces[i].y;//Getting the initial column value of face rectangle's starting point//
        int h = y + faces[i].height;//Calculating the height of the rectangle//
        int w = x + faces[i].width;//Calculating the width of the rectangle//
        rectangle(video_stream, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);//Drawing a rectangle using around the faces//




        vector<Rect> smile;
        smileDetector.detectMultiScale(faceROI, smile, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(80,80));

        if(smile.size() > 0)
        {
            cv::putText(video_stream, "SMILING", Point(x, y), cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118, 185, 0), 2);
        }


      }
      imshow("Face Detection", video_stream);
      //Showing the detected face//
      if( waitKey(10) == 27 || waitKey(10) == 'q' || waitKey(10) == 'Q' ){ //wait time for each frame is 10 milliseconds//
         break;
      }
   }
   return 0;
}


