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
In order to develop a set of images labeled with the correct face shape, I turned to the experts in the fashion and style industry. I will scrape Google images to collect images of each celebrity. Visually inspect to select appropriate images and then organize into folders by shape.
* Review websites (https://www.allure.com/, https://www.marieclaire.com/, https://www.instyle.com/ etc)
* Identify celebrities

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
