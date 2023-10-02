import cv2
import sys
import argparse

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log , cudaFromNumpy
	

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

args = parser.parse_known_args()[0]

# define a video capture object
vid = cv2.VideoCapture(0)

net = detectNet(args.network, sys.argv, args.threshold)

names_list = []
with open('ssd_coco_labels.txt') as f:
    for line in f:
        line = line.replace(" ","")
        line = line.replace("\n","")
        names_list.append(line)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cuda_mem = cudaFromNumpy(frame)

    detections = net.Detect(cuda_mem, overlay=args.overlay)
  
    for detection in detections:

        #Reading bounding box coordinates from detected output
        x1 = int(detection.Left)
        y1 = int(detection.Top) 
        x2 = int(detection.Right)
        y2 = int(detection.Bottom)
        id = detection.ClassID

        #Draw rectangle
        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

        #Annotate class name
        image = cv2.putText(frame,names_list[id] , (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,0,0), 2,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
