# Hair Style Recommendation
----------------
## Hussein Sajid

* [Jupyter Notebook](https://github.com/hussein073/Hair_Style_Recommendation/blob/master/Hair_Style_Recommender.ipynb)
* [Presentation (Google Slides)](https://docs.google.com/presentation/d/1P00kP9RlkvARpfdGx9BnJ7n1cBU9U7WlW1D8v1zjY-0/edit?usp=sharing)
* [Live Web App](http://www.whatsherfaceonline.com)

## Business Understanding
The $20B hair care industry has the opportunity to evolve and differentiate to meet the needs of today’s high-tech, on-the-move women. Hair salons that appeal to these high-earning women must satisfy them by offering a differentiating level of service. With this project, I will attempt to address this challenge by developing a hairstyle recommendation system that identifies the user’s face shape and recommends the most flattering hairstyle.

## Data Understanding
There are five main face shapes often cited on websites and in style magazines. These shapes can be used to find the most flattering hair styles.
* Heart
* Long
* Oval 
* Round 
* Square

## Data Preparation
In order to develop a set of images labeled with the correct face shape, In order to develop a set of images labeled with the correct face shape, I turned to the experts in the fashion and style industry. I reviewed 22 websites and 234 celebrities. Of these, 33 celebrities had a unanimous classification from 3 or more sites (65 from 2 or more). 49 other celebrities had one or two conflicting classifications but had strong consensus towards a face shape with which I could use to classify. As a result, 74 celebrities were utilized in the analysis. There were ~160 celebrities I did not include as they either had too few citations or several conflicts. Because the topic is already subjective, I wanted to ensure I had a level of rigor to labeling my dataset to improve the outcomes. More labeled data would be ideal and would be an area of future enhancement.
* Review websites (https://www.allure.com/, https://www.marieclaire.com/, https://www.instyle.com/ etc)
* Identify celebrities

Of the five face shapes, square faced celebrities were the most agreed upon, with 74% of those celebrities having a unanimous consensus on their shape. Round was second highest at 70%.

Because the classification of face shape is subjective, this will impact the potential accuracy of the model, however, this model attempts to develop an approach to resolving conflict over face shape and providing more stringent guidelines on the definitions.

Square faces have the most consensus which allowed for me to use the most celebrities with square faces.

In order to collect the image dataset, a script was run to download 100 images from Google Images for each celebrity. The images were manually reviewed to ensure they will work for this project (I quickly confirmed that it was an image of celebrity's face and generally face-forward). The appropriate images were saved out to folders with the name of the classified shape. My dataset consisted of ~1500 images for 74 celebrities.

## Modeling
An extraordinary amount of work has been done around computer vision such that a library called face_recognition exists to locate the features of a human face. This library was built using dlib’s state-of-the-art face recognition built with deep learning.

### Feature Extraction
Below is a map of the facial feature map which generates 68 unique points.

<img src="/face_points.jpg" width="600," align="center" style="display: block;margin-left: auto;margin-right: auto;width: 50%;">

Note that the face_recognition package does not provide a top point of the head, so on the basis of a few observations, I determined the half-way point on the face is point 29, so I used double the distance between the chin and point 29 to determine face height. Another available method would be to use the change color (from skin to hair) as the top of the face, however, this would require that the photos not have any bangs and may not work for all skin/hair color combinations.

From these face points, I developed 23 additional features. For the first 16 features, I calculated the angles between the chin point (point 9) and each of the jaw line points (1-8, 10-17). Additionally, I added face width (the distance from point 1 to point 17), face height (described above), the ratio between them, the jaw width (distance between points 7 and 11), the jaw width to face width ratio, the mid-jaw width (distance between points 5 and 13), and the mid-jaw width to jaw width ratio.

My approach will be to first build a facial classifier that will determine whether the user’s face is long, round, oval, heart- or square-shaped. Based on the classification, the model will recommend appealing hairstyles. This classification and recommendation system will help minimize human bias in hair style selection and increase the likelihood that the consumer will be matched with an ideal hair style and therefore more satisfied with her look. I will utilize several techniques and tools from the course: python, visualization, exploratory data analysis, web scraping, feature engineering, featurization, classification models, supervised clustering, unsupervised clustering, artificial neural networks, and possibly TensorFlow and Keras.  I will utilize web scraping to aggregate the celebrities’ pictures and their classified facial shape. Utilizing dlib’s face recognition package, each celebrity’s facial features will be mapped and used to develop a facial shape classification model. The recommendation system will be based on hair styles that are tagged to each face shape. 

From the website user, I will obtain features such as hair type (straight/curly, etc), length preference, and their highest ranking choice. The highest ranking choice can point to the most relevant hair style cluster.

## Evaluation
I will report best hair styles for their face shape on training, test data and recommendation system. Each time they use the system, the application is able to provide feedback that indicates their top and bottom preferences. I plan to use different MLP confusion matrix with test images for further improve the suggested hair styles options based on face shape.

## Deployment
The model will be deployed as an app that allows the user to upload an image of their face, determine their facial shape classification and select and save a gallery of recommended hair styles.

**User Story:**
Meet Jacquelyn She is a new mom and successful journalist. Her company has purchased a table at this year’s USO Gala. With the rare opportunity to have a date night with her husband, Nathan, Jacquelyn wants to look her best. She makes an appointment with a salon on her mobile phone and starts searching online for flattering styles - in between diaper changes and deadlines!
She was confronted with way too many choices, Jacquelyn decides to try the What’s Her Face App.
* Upload Picture
* Run Face Shape Classifier (Oval)
* Recommended Style (Oval - Long) 
Jacqueline decides the beachy waves style will be perfect for the night. 

## Recommender
The recommender python file above contains the code for the recommender which uses as inputs: the face shape classified above, the user's desire for an up-do or not and their hair length.

It returns 6 images of hair styles recommended for their face shape. The original selection is based on a random number score. However, the user can then provide feedback to the system that indicates their top and bottom preference. This preference iterates back into the score by adding or removing points from the score. If the user liked the style, this will increase the priority of the style for the next user and the opposite is true for the user's least favorite style. This is a rudimentary system but utilized as a basic concept.

The recommender images are a subset of the images I used to train the model. I reduced to a subset to remove having too many of the same hairstyles, especially from the same celebrity as well as very outdated hair style.

For your review, I have included the score below each photo. You will see that the style you select as your favorite will increase by 5 points in the next iteration and the one you select as your least favorite will lose 5 points (and may fall off your top 6 list).

There is a lot of opportunity to improve my recommendation system with more time and resources. A few thoughts that I would explore in the future:

1) A recommendation system that employs colloborative filtering. The CF algorithm works by identifying a set of users with similar past preferences as the target user's. The algorithm recommends N items to the target user based on what the similar people like. In this case, there is no consideration of the items' characteristics themselves.

This recommendation system can be improved by allowing users to set up profiles that collect information that can be used to create these similar sets of users (in addition to their past preferences). This would help the system identify a new user's possibly set of similar users because it provides some information upfront about the user before they begin to rate the items in the system. I think this would work well for my project because there are additional attributes that may impact someone's choice of hairstyle (other than length and face shape). For example, hair texture, volume, elasticity and color as well as age and skintone, may impact the user's hair style preference. This could be collected upfront (since it is less likely to change) and people with similar profiles could be utilized for ranking style recommendations.

2) Content-based recommendation system. This algorithm focuses on the features of the items you are recommending. This requires that the items have some sort of metadata or characteristics. This system recommends items based on their similarity to other items the user has liked. This is more difficult for this project because the tagging of the hair styles would be relatively manual and subjective. Unlike movie titles, software, articles, books, etc that have more readily-available metadata, the hair styles would be difficult to utilize a content-based recommendation system.

Ideally, the recommendation system would represent a hybrid of both colloborative filtering and content-based recommendation algorithsm. The former is easier to implement and can still produce strong results. My recommendation system below is used to illustrate that the recommendations are based on the algorithm above to find face shape as well as a machine-learning algorithm that begins to learn preferences and store scores for each style (which could be associated to a user within an app or website). The possibilities are endless and the user would be able to rely upon the algorithms to give them the answers to their pressing style questions with little subjectivity and access to matching themselves up with styles from the best hair stylists in the industry.

## Requirements

* dlib

```
pip install dlib
```
* CMake

```
pip install cmake
```
* Face Recognition

```
pip install face_recognition
```
## Running the recommendation system
Clone, and cd into the repo directory. The first thing you need to do is to get the required packages installed by running above commands in the terminal or append '!' before command in the jupyter notebook.

Note: For security reasons browsers do not allow to retrieve full file path in browser and hence no access to the File System. When a file is selected by using the input type=file object, the value of the value property depends on the value of the "Include local directory path when uploading files to a server" security setting for the security zone used to display the Web page containing the input object.

The fully qualified filename of the selected file is returned only when this setting is enabled. When the setting is disabled, Browser replaces the local drive and directory path with the string C:\fakepath\ in order to prevent inappropriate information disclosure. 

You should save the recommended image in the folder (data -> pics -> recommendation_pics) of the repository. To run the flask app:
```
FLASK_APP=app.py flask run
```
