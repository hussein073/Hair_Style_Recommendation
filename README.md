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

## Requirements
* Face Recognition

```
pip install face_recognition
```
