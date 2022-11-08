# AML-for-Image-Analysis

### IrisLocalization

The approach we took for iris localization follows a series of logical steps:

- We localized the pupil. The pupil is the easiest part of the eye to extract 
as it consistently has a sharp edge an realtively similar sizes. Also if the 
quality of the images are good, it is almost always unobstructed by eyelids,
unlike the iris. 
  - First we applied a gaussian blur to the image, then we used edge detection
  and a houghes circles algorithm to detect circles. We put realtively strict 
  set of parameterrs on houghes circles so that it only selects the strongest 
  candidate of a circle in the image.
  - Note: we did not equalize these image because a pupil is already almost always 
  completely black, so equalizing the image would actually bring the surrounding
  colors closer to that of the pupil making is more difficult to detect.
- From here we narrowed the search for the iris. To do this we created a mask of the
image removing the pupil and three times the pupil's radius from consideration. 
We found that typically, a radius of three times that of a pupil was large enough 
to extend beyond the iris making it a justifiable mask for the image without 
removing the iris edge.
  - In the preprocessing for iris detection we equalize the images as lighter color 
  iris's are harder to detect without this step.
  - We then repeated the edge detection steps on this narrowed down image with one 
  big difference. The radius of the iris has to be larger than that of the pupil and 
  smaller than that of the outside bounding edge of three times the pupil radius,
  so the minimum and maximum for the radius is constrained by these factors.
- From here we set the pupil and iris coordinates with one caveat, for the sake
of normalization, we need an equal width ring for the iris, therefore we have centered
the iris radius around the pupil's center. Without this the normalization process
is problematic
- We check that the iris ring is not extending beyond the borders of the 
image. We do this by calculating the shortest distance between the center of the
eye (the pupil center) and the edge of the image. If this distance is ever exceeded 
by the radius of the iris from the center of the eye, then we exchange the radius 
withe one bounded to fit inside the image. Again this step is for proper 
normalization.
- Finally, we mask the image according to the pupil and iris. Then pass it on 
to normalization

### IrisNormalization

For iris Normalization we follow a simple protocol to ensure standardized outputs for
comparison:

- First we center the image around the center of the pupil with the boundaries of 
the image lying at the edge of the iris. In other words, the image is transformed 
to a square with edge length the size of the iris diameter.
- From here we "unroll" the iris with Daugman's rubber sheet method. This method
converts polar to cartesian coordinates. When this is applied to the entire image you
get a square image with the pupil radius of black space above the rectangle iris,
and a below the iris more black space for the background so we crop the image to
consist of only the iris rectangle.
- We then pass of this normalized rectangle to iris enhancement.

### IrisEnhancement

In Iris Enhancement we are looking to remove the noise in the normalized image
and make the patterns more distinguishable.

- First we have to remove the eyelids as they often cover part of the iris and 
are considered noise. To do this we first apply a gaussian blur to remove the 
texture noise from the image.
- We then run the houghes circles algorithm to detect the parabolic edges 
that the eyelids have in the normalized image. The radius parameters are confined
to ensure no small circle noise is detected and to limit the amount that can 
be removed from the image without limiting its ability to detect the eyelid
parabola. We also set a smaller min distance (about 1/3 of the image length)
to allow for more than one eyelid to be detected in the cases where both eyelids
block the iris.
- We then mask of the eyelid noise based on the circles detected
- From here we equalize the image to enhance the patterns and aid in feature 
extraction. Then we standardize the size so that there is a set number of values
to extract and compare.

### FeatureExtraction

Ethan

### IrisMatching

There are several steps between feature extraction and fitting the machine learning
model for iris matching:

- First we have to iterate through the database and get paths for each individual image. 
- Our training and test set were predefined so there was no need to split the data.
- Once we have our train and test set we split them into X and y sets (y being the  
target feature and X being the feature matrix).
- Since there are so many values to be considered (1536), we use Linear Discriminant analysis
to reduce the dimensionality of the data. This works by finding the most strongly correlated
features and reducing the number of features to reflect the non-correlated groups. We then
apply the dimensionality reduction to the test set as well.
- One we have a dataframe standardized, we can go ahead and fit our nearest centroid classifier
to our training data. Once the mode is trained we apply the model to the incoming iris's (the
test set) to determine which individual they belong to. 
- From our predictions we are then able to evaluate the model.

### PerformanceEvaluation

The main indicator of this model is going to be accuracy, as this is a matching program
not a authentication one. If it were an authentication we would look to limit the false
positives as much as possible. But since this is just a matching program, we want to see
how many people we can correctly match to their iris. In this case we have accuracy of around
80% which means 80% of the test set was correctly matched to the person for which the iris 
belongs. This is our best metric for model performance.

The Reciever Operator Characteristic (ROC) allows us to see whether we are getting false 
positives or true negatives as our error if this were a binary class. However, we can use 
an average of the "one versus the rest" method of classification to determine the performance
of the model.

### Limitations
