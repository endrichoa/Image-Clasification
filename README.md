# Machine Learning Project Report - Endricho Abednego

## Project Domain

*Pistachio (Pistacia Vera L.)* is an agricultural product originating from the Middle East and Central Asia. There are two main types of pistachios typically grown and exported from their country of origin, Turkey: *Kirmizi Pistachios* and *Siirt Pistachios*. Each type has a different price, taste, and nutritional value. Therefore, a *machine learning* model is needed to perform image classification between these two types of pistachios, namely Kirmizi and Siirt.

Additionally, it is important to determine the quality of the pistachios with high accuracy and ease through the machine learning model that will be created. The main goal of creating this model is to classify pistachio types using images in a fast and effective manner.

There is a study that classifies these two types of pistachios using 2148 images, divided into 1232 images of Kirmizi pistachios and 916 images of Siirt pistachios, where each image is 600 x 600 *pixels* in size.

## Business Understanding

This project is built for a community with the following business characteristics:
+ Turkish people who work as Pistachio sellers so they can distinguish between the two types of pistachios mentioned above.
+ Pistachio consumers so they can understand the differences between the two types in terms of price, taste, and nutrition.

### Problem Statements

- How to create a machine learning model to classify images of two types of pistachios with high accuracy?
- What features are contained in the identification of pistachios?
- What is the best algorithmic method for performing pistachio classification based on the existing dataset?

### Goals
- To create a machine learning model by determining the most effective algorithm to solve the pistachio classification problem.
- To identify what features exist in pistachio identification.
- To analyze the existing dataset to determine which algorithm is suitable for building the model.

### Solution statements
- Understand the dataset to be used for building this project by performing data visualization.
- Find the appropriate algorithm or model based on the dataset to be used.
- Use 2 types of models for this project to directly test the models and compare which one is better and more efficient for this project.
- Use the *Accuracy* evaluation metric as a benchmark in comparing the 2 models used.

## Data Understanding
The dataset used is the *Pistachio Image Dataset* with two classification types. This dataset was taken from Kaggle and can be downloaded via the following link [Pistachio Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset).

### Classes in the dataset
There are two classes in the dataset, namely:
+ Pistachio Kirmizi
+ Pistachio Siirt

### Detailed information regarding the dataset:
+ The dataset has the format (.jpg).
+ The dataset contains 2148 images.
+ The dataset is divided into a *Training Set* and *Validation Set* with a ratio of (80/20).

### Data Visualization

Below is a data visualization that can be used to help understand the dataset that will be used in this project.
![Data Sample](https://github.com/endrichoabednego/Dicoding-Academy/blob/main/GambarTerapan1/WhatsApp%20Image%202022-12-03%20at%2011.14.25%20PM.jpeg?raw=true)
Figure 1. Data Sample

Based on Figure 1 above, it can be seen that the dataset to be used features two types of pistachios: Kirmizi Pistachio and Siirt Pistachio.

## Data Preparation
+ Download Dataset
Before downloading the dataset, it is necessary to import the dataset in Kaggle via the link above using the Kaggle API by uploading the kaggle.json file as an API key into the notebook.

+ *Zip Extraction*
After successfully downloading the dataset, file extraction is still required because the downloaded file is in .zip format. Once extracted, the dataset can be used.

+ *Split Data*
In this project, the dataset will be divided into a *train set* and *validation set* with a ratio of (80/20). To perform this data split, it is necessary to import the *splitfolders library*, but before importing, it is also necessary to install the *library*.

+ *Labeling Data*
After performing the *split data*, the next step is data *labeling*. For this project, the data is divided into 2 types: Kirmizi and Siirt. Dataset *labeling* is required to classify the pistachio variant types obtained through the dataset.

## Modeling
1. *Image Augmentation*
In this process, image augmentation will be performed, which functions to artificially increase the size of the image dataset; this can be achieved by applying random transformations to the images. This project uses ImageDataGenerator to perform image augmentation, which is useful for duplicating images and adding variations according to the added functions. There are several variables used to perform variations in this project, including:
   - *rescale*
   This process is used to normalize every pixel value in the image to a value between 0 and 1.
    - *horizontal_flip*
    This process functions to flip the image horizontally, and in this project, it has a *value* of *True*.
    - *shear_range*
    The image will be slanted along a specific axis to create or correct the angle of perception; in this project, a *value* of 0.2 is given.
    - *zoom_range*
    As the name implies, it performs augmentation in the form of a *zoom* on the image by a specified *value*, which in this case is 0.2.
    - *rotation_range*
    Performs random rotation on the image; in this project, a *value* of 20 is given.
    -  *width_shift_range* and *height_shift_range*
    Performs shifting of the width and height of the image; in this case, both variables are given a *value* of 0.2 for both *height* and *width*.
    - *vertical_flip*
    This process functions to flip the image vertically, and in this project, it has a *value* of *True*.
    -  *fill_mode*
    When a pixel has an empty value, the nearest pixel will be chosen and repeated to fill all those empty pixel values.

2. Flow Train Set Data and Validation Set Data
This project also uses the *flow_from_directory()* function which is useful for taking the augmented data to be loaded into memory. There are target_size, batch_size, and class_mode variables; these variables function to provide limits on the data for both the train and validation sets.

3. *Model Building*
This project uses two types of algorithms for the models being built: the first uses a base model and the second uses VGG16.
  - *Base Model*
For the base model, it begins by using a Conv2D Maxpooling Layer used for modeling in *machine learning* projects based on digital images. The layer used for this project is *sequential*; the arrangement of this layer begins by providing a complex convolution layer with a relu *activation* function. Then, *MaxPool2D* will be performed with a *pool_size* of (2,2) which will equally divide the input from each spatial dimension. Then, at the end of the *layer*, a Dense layer function with a *softmax activation* function will be provided as the *output layer*. Once the *layer* is built, it will enter the *compile* process using the *adam optimizer* with the *categorical_crossentropy loss function*.
The evaluation metric used for this base model is *accuracy*.

- *VGG16 model*
To build the VGG16 model, it is necessary to *import the VGG16 application*.
After successfully importing, the *weight* parameter will be used to determine the weight checkpoint from which the model is initialized. After that, there is *include_top*; by default, the classifier will be connected according to the 1000 classes from *ImageNet*, but so that the previous *network layer* is not connected to the model, the value will be set to *False*. Finally, there is *input_shape* which is the image tensor fed into the network.
The construction of the *layer* on the VGG16 model will begin with *GlobalAveragePooling* to replace the *fully connected layer* for classification in this project. Then at the end of the *layer*, a *Dense layer* function with a *softmax activation* function will be provided as the *output* layer. Then the input layer process into model_vgg16 is performed. After the layer is successfully built and inserted into model_vgg16, it will enter the *compile* process using the *adam optimizer* with the *categorical_crossentropy loss function*. The metric used for this model is *accuracy*.

## Evaluation
In this project, the evaluation metric used is *accuracy*. As the name implies, this evaluation metric can determine the accuracy of the prediction results with the original data from the *train and validation set*. The measurement scale of this metric is between 0 and 1, meaning that the higher the *accuracy* or the closer it is to the number 1, the better the prediction results from the model, and the lower the *accuracy* or the closer it is to the number 0, the worse the prediction results from the model.

![Base Model Accuracy](https://github.com/endrichoabednego/Dicoding-Academy/blob/main/GambarTerapan1/akurasibase.jpeg?raw=true)
Figure 2. Base Model Accuracy Graph

![Base Loss Plot](https://github.com/endrichoabednego/Dicoding-Academy/blob/main/GambarTerapan1/lossbase.jpeg?raw=true)
Figure 3. Base Model Loss Plot Graph

It can be seen from the two figures above that, broadly speaking, the accuracy of the base model created in this project increased and obtained the highest accuracy value of 0.9250. Meanwhile, the base model loss plot decreased and obtained the lowest plot loss value of 0.1935.

![VGG16 Accuracy](https://github.com/endrichoabednego/Dicoding-Academy/blob/main/GambarTerapan1/akurasivgg16.jpeg?raw=true)
Figure 4. VGG16 Model Accuracy Graph

![VGG16 Loss Plot](https://github.com/endrichoabednego/Dicoding-Academy/blob/main/GambarTerapan1/lossvgg16.jpeg?raw=true)
Figure 5. VGG16 Model Loss Plot Graph

For the VGG16 method, a slight increase in accuracy occurred, obtaining the highest accuracy of 0.93271, while the loss also experienced a decrease to the figure of 0.1942.

Based on the graphs and results obtained from the two algorithms above, it can be concluded that VGG16 has higher accuracy and lower loss. Therefore, for the dataset used in this project, the VGG16 algorithm is a better solution compared to the base model.

## Conclusion
Based on the *accuracy* results obtained through the two models that have been created, it can be concluded that by using the *Pistachio Image Dataset*, the model with the VGG16 *application* is a more effective model and a better solution compared to the *base* model.

## References

[1] SINGH D, TASPINAR YS, KURSUN R, CINAR I, KOKLU M, OZKAN IA, LEE H-N., (2022). Classification and Analysis of Pistachio Species with Pre-Trained Deep Learning Models, Electronics, 11 (7), 981.
