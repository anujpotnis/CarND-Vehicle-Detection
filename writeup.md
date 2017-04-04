##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./writeup_data/car_heat1.png
[image2]: ./writeup_data/car_heat2.png
[image3]: ./writeup_data/car_heat3.png
[image4]: ./writeup_data/car_heat4.png
[image5]: ./writeup_data/car_heat5.png
[image6]: ./writeup_data/car_heat6.png
[image7]: ./writeup_data/bheat.png
[image8]: ./writeup_data/aheat.png
[image9]: ./writeup_data/car_nocar.png
[image10]: ./writeup_data/car_nocar2.png
[image11]: ./writeup_data/car_nocar3.png
[image12]: ./writeup_data/car_nocar4.png
[image13]: ./writeup_data/normalize.png
[image14]: ./writeup_data/sliding.png
[image15]: ./writeup_data/car_hog1.png
[image16]: ./writeup_data/car_hog2.png
[image17]: ./writeup_data/car_hog3.png
[image18]: ./writeup_data/car_hog4.png
[image19]: ./writeup_data/normalize.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Please find the writeup.md file.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG features were extracted using the skimage function 'hog'. Parameters for tuning such as pixels per cell, cells per block, number of histograms, orientations were passed to the hog function. The 'visualize' parameter allowed me to visualize the hog features next to an image. The 'feature_vector' reshaped the hog features to a 1-D vector.

```python
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]




####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Increasing the number of histogram bins and orientations increased accuracy but took more time to compute. Larger cells were faster but less accurate. Finally the following combination was a balance between the computation and accuracy:

```python
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
```



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The features of the cars and no-cars were stacked and accordingly even the corresponding labels were stacked.

One of the most important operation performed on the data was that of normalization. Every type of feature has output of varying either from 0-1 or from 0-255. It is important that all of them are normalized. Also, the same normalization operation must be performed on the test feature vector.

![alt text][image19]

The data was split between 80% training and 20% testing. This was important to gause the accuracy of the model.

Finally, a Linear SVM was used to train.

```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)
```

```python
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
21.45 Seconds to train SVC...
Test Accuracy of SVC =  0.9865
```



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Search for smaller cars was restricted in the upper portion of the images. Scaling allowed the cars to be detected even after they were far. The final list of the bounding boxes was then passed to the heatmap.

```python
scale_list = [0.6, 1, 1.5, 2]
ystop_list = [500, 500, 530, 700]
bboxes_scale = []
    
    for scale, ystop in zip(scale_list, ystop_list):
        out_img, bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        bboxes_scale = bboxes_scale + bbox_list
    heat = np.zeros_like(out_img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,bboxes_scale)
```



![alt text][image14]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

A combination of features was important for the successful detection of vehicles.

For example, in the first image, the colors of the cars are dominant. Hence the color based features help here. In the second image, the black car is in the shade of the tree. Here the HOG features play an important role.

In the third image we can see how the the sliding windows (with overlap) help increase the confidence of a detection. The heat map at this point is 'hot' which after thresholding detects the car with higher confidence.

![alt text][image4]
![alt text][image5]
![alt text][image7]

In order to optimize the classifier, the HOG was calculated for the entire image only once and them sub images were sliced out of it. This significantly saved computation.

```python
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
```



---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

A global heatmap was implemented to capture heatmaps across different frames. This was implemented using the 'deque'

```python
MAXLEN=5
global_heatmap = deque(maxlen=MAXLEN)
```


Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` (actually the output generated using label() ) on the integrated heatmap from all six frames:
![alt text][image7]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image8]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Extracting the right features was challenging. Also choosing the right scale for the right distance is a result of trial and error.
The pipeline can fail in low light since only the tail lights are visible. A human can interpret the tail lights to be that of a car but for a system trained on hog features, it can fail.
More than one SVM model is required for day and night.

