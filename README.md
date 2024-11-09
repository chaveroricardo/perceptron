# Binary Classifier for In-Vivo Microscopic Images for Esophageal Cancer Diagnosis

A linear Perceptron is used to perform binary classification on a subset of images from the [Data Challenge by Mauna Kea](https://challengedata.ens.fr/participants/challenges/11/).

For this classifier, only images of healthy tissue and images of tissue with dysplasia/cancer were used. Thus, the dataset consists of 1,469 images of healthy tissue (class 0) and 3,594 images of dysplasia/cancer tissue (class 1).

The original images were scaled from 519x521 pixels to 260x260 to reduce the time and memory required for processing. The set of images used is available in the compressed file [ImageFolder.zip](https://drive.google.com/file/d/1Abi4hjl5djn8X75YCcMXL5htq7iqf7VY/view?usp=sharing), outside this repository.

Additionally, the file *ClasesImagenes.csv*, located in the *Data* folder of this repository, contains a table identifying the name of each image and the class to which it belongs.
