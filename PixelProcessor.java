package org.firstinspires.ftc.teamcode;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import android.graphics.Canvas;

// All VisionProcessor interfaces expect/have some init function, processFrame function, and onDrawFrame function
public class PixelProcessor implements VisionProcessor {
    /*
    *   PixelProcessor is a subclass of VisionProcessor used to detect the color of "intaked" pixels in FTC Center Stage.  
    *   
    *   Parameters: 
    *      - width (int): width, in pixels, of the camera stream  
    *      - height (int): height, in pixels of the camera stream
    *      - calibration (CameraCalibration object): input camera stream
    *   Methods: 
    *      - processFrame -> Executed every time a new frame is dispatched from input source, 
           returns null (typically the image that will be displayed in the viewport)
    *      - onDrawFrame -> used for drawing annotations onto the displayed image; not used in this implementation
    */

    // Declare variables to store OpenCV Rectangle objects 
    Rect TOP_DETECTION_REGION;
    Rect BOTTOM_DETECTION_REGION; 
    Rect DETECTION_REGION; 

    // Declare variables and assign to instantiated matrix 
    Mat hsvMat = new Mat();             // represents a HSV-converted image frame 
    Mat detectedMat = new Mat();        // stores the processed image frame 

    // Green HSV Range
    Scalar GreenThresholdLow = new Scalar(35, 100,20);
    Scalar GreenThresholdHigh = new Scalar(78,255,255); 
    // Blue HSV Range
    Scalar BlueThresholdLow = new Scalar(93, 100,20);
    Scalar BlueThresholdHigh = new Scalar(138,255,255);
    // Yellow HSV Range
    Scalar YellowThresholdLow = new Scalar(20, 100,20);
    Scalar YellowThresholdHigh = new Scalar(33,255,255);
    // White HSV Range
    Scalar WhiteThresholdLow = new Scalar(15, 10, 200); 
    Scalar WhiteThresholdHigh = new Scalar(25, 30, 255);
    // 2D array for storing the lower and upper bounds of each color 
    Scalar[][] ColorThresholds = {
                            {GreenThresholdLow, GreenThresholdHigh}, 
                            {BlueThresholdLow, BlueThresholdHigh}, 
                            {YellowThresholdLow, YellowThresholdHigh},
                            {WhiteThresholdLow, WhiteThresholdHigh}
                        }; 

    // Thresholds for white pixel concentration for each region 
    double topThreshold = 0.01;
    double bottomThreshold = 0.01;

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
        /* 
        * Runs once on initialization. Assigns width, height values to the rectangle objects once the camera stream is on. 
        */

        // Top left of the frame is (0,0) --> bottom right corner is (width, height) 
        // Takes in two opposite corners of the desired rectangle, in this case top left and bottom right 
        // Instantiate Rectangle objects representing the detection area for the top and bottom "intaken" pixels
        BOTTOM_DETECTION_REGION = new Rect(
            new Point(0.16*width, 0.425*height),
            new Point(0.85*width, 0.8*height)
        );
        TOP_DETECTION_REGION = new Rect(
            new Point(0.16*width, 0.07*height),
            new Point(0.85*width, 0.425*height)
        );
    }
    
    @Override
    public Object processFrame (Mat frame, long captureTimeNanos) {
        /*
         * Runs repeatedly when the camera stream is on, processing the video frame by frame
         * 
         * width: pixel width of camera frame 
         * height: pixel height of camera frame
         * captureTimeNanos: elapsed time in nanoseconds
         * 
         * returns null in this implementation because it interfaces with EOCVSim, 
         * typically returns the image matrix that will be displayed in the viewport
         */

        // Convert frame matrix from RGB to HSV representation and store in hsvMat 
        Imgproc.cvtColor(frame, hsvMat, Imgproc.COLOR_RGB2HSV);

        // Represents the color of the red and green borders in RGB
        Scalar redBorder = new Scalar(255,0,0);
        Scalar greenBorder = new Scalar(0, 255, 0);
        Scalar yellowBorder = new Scalar(255, 255, 0);
        Scalar whiteBorder = new Scalar(255, 255, 255);
        Scalar blueBorder = new Scalar(0, 0, 255);
        // Array storing border color Scalar objects 
        Scalar[] BorderColors = {greenBorder, blueBorder, yellowBorder, whiteBorder}; 
        
        // Loops through each HSV color range in the ColorThresholds array, extracting the lower and upper Scalars
        // Applies the color to do colored pixel detection in the TOP REGION
        // Continues iterating until the color of the is found or end of loop (if there is no pixel) 
        for (int i = 0; i < 4; i++) {
            // All pixels that satisfy the threshold range are stored as white pixels and all other pixels as black pixels in detectedMat
            Core.inRange(hsvMat, ColorThresholds[i][0], ColorThresholds[i][1], detectedMat);

            // Calculate the percentage of white pixels (pixels in the color threshold range) in the image matrix
            double TopPercent = (Core.sumElems(detectedMat.submat(TOP_DETECTION_REGION)).val[0] / 255) / TOP_DETECTION_REGION.area();

            // If the percentage of white pixels exceeds the threshold, then there is a pixel of the selected color in the top region
            if (TopPercent > topThreshold) {
                // Draw top bounding box rectangle and color it correspondingly  
                Imgproc.rectangle(frame, TOP_DETECTION_REGION, BorderColors[i]);
                // Stop searching for color because it has been found, breaking the loop
                break; 
            // Otherwise, there is either no pixel in the detection region or a pixel of a different color; keep searching
            } else {
                // Draw top bounding box rectangle and color it red  
                Imgproc.rectangle(frame, TOP_DETECTION_REGION, redBorder);
            }
        }

        // Loops through each HSV color range in the ColorThresholds array, extracting the lower and upper Scalars
        // Applies the color to do colored pixel detection in the BOTTOM REGION
        // Continues iterating until the color of the is found or end of loop (if there is no pixel) 
        for (int i = 0; i < 4; i++) {
             // All pixels that satisfy the threshold range are stored as white pixels and all other pixels as black pixels in detectedMat
            Core.inRange(hsvMat, ColorThresholds[i][0], ColorThresholds[i][1], detectedMat);

            // Calculate the percentage of white pixels (pixels in the color threshold range) in the image matrix
            double BottomPercent = (Core.sumElems(detectedMat.submat(BOTTOM_DETECTION_REGION)).val[0] / 255) / BOTTOM_DETECTION_REGION.area();

            // If the percentage of white pixels exceeds the threshold, then there is a pixel of the selected color in the bottom region
            if (BottomPercent > bottomThreshold) {
                // Draw bottom bounding box rectangle and color it correspondingly  
                Imgproc.rectangle(frame, BOTTOM_DETECTION_REGION, BorderColors[i]);
                // Stop searching for color because it has been found, breaking the loop
                break; 
            // Otherwise, there is either no pixel in the detection region or a pixel of a different color; keep searching
            } else {
                // Draw bottom bounding box rectangle and color it red  
                Imgproc.rectangle(frame, BOTTOM_DETECTION_REGION, redBorder);
            }
        }
        return null;
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        /*
         * This method is used for drawing annotations onto
         * the displayed image, e.g outlining and indicating which objects
         * are being detected on the screen, using a GPU and high quality 
         * graphics Canvas which allow for crisp quality shapes
         * 
         * Not used in this implementation
         */
    } 
}
