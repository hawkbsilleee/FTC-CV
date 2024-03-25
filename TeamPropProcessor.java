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
public class TeamPropProcessor implements VisionProcessor {
    /*
    *   TeamPropProcessor is a subclass of VisionProcessor used to detect #14712's red team props for FTC Center Stage.  
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

    // Declare variables for OpenCV Rectangle objects 
    Rect LEFT_RECTANGLE;
    Rect MIDDLE_RECTANGLE;
    Rect RIGHT_RECTANGLE;

    // Declare variables and assign to instantiated matrix 
    Mat hsvMat = new Mat();             // represents a HSV-converted image frame 
    Mat lowMat = new Mat();             // stores pixels that satisfy/don't satisfy the lower threshold
    Mat highMat = new Mat();            // stores pixels that satisfy/don't satisfy the upper threshold
    Mat detectedMat = new Mat();        // stores the combined upper and lower thresholds 

    /* 
    * RGB vs. HSV Explanation:
    *   RGB (red, green, blue) represents a mix of three primary colors. So, when you're trying to detect 
    *   the color of some object that also varies in different environments due to factors like lighting, its 
    *   difficult to arbitrarily dictate how much of each primary color composes it. 
    *   HSV (hue, saturation, value) functions in a way were you can represent the hue (color) separate from 
    *   the saturation and value, which mimick different lighting conditions, making it ideal for thresholding. 
    *   The following ranges represent the HSV spectrum: 
    *       - Hue: [0, 360] in degrees
            - Saturation: [0, 255] or as a percentage 
            - Value: [0, 255] or as a percentage
    *   Note: OpenCV uses 0-180 instead of 0-360 for hue so take half of the value 
    */

    // Instantiate scalar objects that represent the red detection threshold in the lower red spectrum
    // min hue, sat, val 
    Scalar lowerRedThresholdLow = new Scalar(0, 100, 20); 
    // max hue, sat, val
    Scalar lowerRedThresholdHigh = new Scalar(10, 255, 255); 

    // Instantiate scalar objects that represent the red detection threshold in the upper red spectrum
    // min hue, sat, val 
    Scalar upperRedThresholdLow = new Scalar(160, 100, 20);
    // max hue, sat, val
    Scalar upperRedThresholdHigh = new Scalar(179, 255, 255);


    // Thresholds for white pixel concentration for each region 
    double leftThreshold = 0.01;
    double middleThreshold = 0.01;
    double rightThreshold = 0.01;

    // Create a PropLocation object and set the location attribute to no found 
    PropLocation propLocation = PropLocation.NOT_FOUND; 


    @Override
    public void init(int width, int height, CameraCalibration calibration) {
        /* 
        * Runs once on initialization. Assigns width, height values to the rectangle objects once the camera stream is on. 
        */

        // Top left of the frame is (0,0) --> bottom right corner is (width, height) 
        // Takes in two opposite corners of the desired rectangle, in this case top left and bottom right 
        // Instantiate Rectangle object representing the detection area for the left spike mark  
        LEFT_RECTANGLE = new Rect(
            new Point(0, 0.286 * height),
            new Point(0.33 * width, 0.66*height)
        );
        // Instantiate Rectangle object representing the detection area for the middle spike mark  
        MIDDLE_RECTANGLE = new Rect(
            new Point(0.33 * width, 0.286 * height),
            new Point(0.66 * width, 0.66*height)
        );
        // Instantiate Rectangle object representing the detection area for the right spike mark  
        RIGHT_RECTANGLE = new Rect(
            new Point(0.66*width, 0.286 * height),
            new Point(width,0.66*height)
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
         * typically the image that will be displayed in the viewport
         */

        // Convert frame matrix from RGB to HSV representation and store in hsvMat 
        Imgproc.cvtColor(frame, hsvMat, Imgproc.COLOR_RGB2HSV);
        
        // Represent all pixels in hsvMat in the red threshold range (lower spectrum) as white pixels and all other pixels as black pixels in lowMat   
        Core.inRange(hsvMat, lowerRedThresholdLow, lowerRedThresholdHigh, lowMat);
        // Represent all pixels in hsvMat in the red threshold range (high spectrum) as white pixels and all other pixels as black pixels in highMat   
        Core.inRange(hsvMat, upperRedThresholdLow, upperRedThresholdHigh, highMat);

        // Combines lowMat and highMat in detectedMat to represent all pixels that satisfy the thresholds
        // For a given pixel location in lowMat and highMat: if both pixels are white, the corresponding pixel in detectedMat will be white; 
        // if either pixel is white, the corresponding pixel will be white; if neither of the pixels are white (both black), the corresponding pixel will be black
        Core.bitwise_or(lowMat, highMat, detectedMat);

        // Submat function returns a specified portion of detectMat 
        // SumElems takes all the pixel values and adds them together (black = 0, white = 255) --> higher number means more white, lower number means more black 
        // --> more white means object satisfies the threshold better 
        // So, Core.sumElems(detectedMat.submat(LEFT_RECTANGLE)).val[0] / 255 returns how many white pixels there are in the specified area 
        // and dividing this value by the area of the specified area gives the proportion of white pixels in the specified area 
        double leftPercent = (Core.sumElems(detectedMat.submat(LEFT_RECTANGLE)).val[0] / 255) / LEFT_RECTANGLE.area();
        double middlePercent = (Core.sumElems(detectedMat.submat(MIDDLE_RECTANGLE)).val[0] / 255) / MIDDLE_RECTANGLE.area();
        double rightPercent = (Core.sumElems(detectedMat.submat(RIGHT_RECTANGLE)).val[0] / 255) / RIGHT_RECTANGLE.area();

        // Represents the color of the red and green borders
        Scalar redBorder = new Scalar(255,0,0);
        Scalar greenBorder = new Scalar(0, 255, 0);

        // If left region has the highest concentration of white pixels and is above the threshold, team prop is detected in left region
        if (leftPercent > middlePercent && leftPercent > rightPercent && leftPercent > leftThreshold) {
            // Set the team prop location attribute to left 
            propLocation = PropLocation.LEFT; 
            // Draw three rectangles representing the left, right, and middle detection regions 
            // Color the boundary of the left region green and the others red
            Imgproc.rectangle(frame, LEFT_RECTANGLE, greenBorder);
            Imgproc.rectangle(frame, MIDDLE_RECTANGLE, redBorder); 
            Imgproc.rectangle(frame, RIGHT_RECTANGLE, redBorder); 
        // If middle region has the highest concentration of white pixels and is above the threshold, team prop is detected in middle region
        } else if (middlePercent > leftPercent && middlePercent > rightPercent && middlePercent > middleThreshold) {
            // Set the team prop location attribute to middle
            propLocation = PropLocation.MIDDLE;
            // Draw three rectangles representing the left, right, and middle detection regions 
            // Color the boundary of the middle region green and the others red
            Imgproc.rectangle(frame, LEFT_RECTANGLE, redBorder);
            Imgproc.rectangle(frame, MIDDLE_RECTANGLE, greenBorder); 
            Imgproc.rectangle(frame, RIGHT_RECTANGLE, redBorder); 
        // If right region has the highest concentration of white pixels and is above the threshold, team prop is detected in right region
        } else if (rightPercent > leftPercent && rightPercent > middlePercent && rightPercent > rightThreshold) {
            // Set the team prop location attribute to right
            propLocation = PropLocation.RIGHT;
            // Draw three rectangles representing the left, right, and middle detection regions 
            // Color the boundary of the right region green and the others red
            Imgproc.rectangle(frame, LEFT_RECTANGLE, redBorder);
            Imgproc.rectangle(frame, MIDDLE_RECTANGLE, redBorder); 
            Imgproc.rectangle(frame, RIGHT_RECTANGLE, greenBorder); 
        // If no region has a high concentration of white pixels that is above the threshold, no team prop is detected
        } else {
            // Set the team prop location attribute to not found 
            propLocation = PropLocation.NOT_FOUND;
            // Draw three rectangles representing the left, right, and middle detection regions 
            // Color all three regions' boundary red
            Imgproc.rectangle(frame, LEFT_RECTANGLE, redBorder);
            Imgproc.rectangle(frame, MIDDLE_RECTANGLE, redBorder); 
            Imgproc.rectangle(frame, RIGHT_RECTANGLE, redBorder); 
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

    // Getter function that returns the location attribute of the PropLocation object
    public PropLocation getPropLocation() {
        return propLocation; 
    }

    // An enum is a special class the represents a group of constants (final variables) 
    // Each enum constant is initialized with an integer value 
    public enum PropLocation {
        LEFT(1), 
        MIDDLE(2),
        RIGHT(3), 
        NOT_FOUND(0); 

        public final int posNum;
        
        // Constructor which takes an integer parameter posNum, used to assign the integer value to the posNum field of each enum constant
        PropLocation(int posNum) {
            this.posNum = posNum; 
        }
    } 
}
