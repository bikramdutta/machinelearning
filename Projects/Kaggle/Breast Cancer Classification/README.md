# Breast Cancer Tumour Classification
*Project for a Kaggle Competition.*
*One of the most important case studies for conceptual learning.*

## Project Overview
- Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
- 30 features are used, examples:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry 
        - fractal dimension ("coastline approximation" - 1)

- Datasets are linearly separable using all 30 input features
- Number of Instances: 569
- Class Distribution: 212 Malignant, 357 Benign
- Target class:
         - Malignant
         - Benign

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

![image.png](attachment:image.png)

## Important steps involved
1. Import the data from sklean datasets
2. Normalizing the data :: Range of (0,1)
3. Examine the data and observe the correlations between features
4. Visualize using pairplot 
5. Take decision on the model to be used
6. SVC model used 
7. Accuracy observed

## Result
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.96   | 0.98     | 48      |
| 1.0          | 0.97      | 1.00   | 0.99     | 66      |
| accuracy     |           |        | 0.98     | 114     |
| macro avg    | 0.99      | 0.98   | 0.98     | 114     |
| weighted avg | 0.98      | 0.98   | 0.98     | 114     |

## Software and Libraries

This project uses the following software and Python libraries:

* [Python](https://www.python.org/downloads/release/python-364/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/0.17/install.html)
* [NumPy](http://www.numpy.org/)
* [Seaborn](https://seaborn.pydata.org/)










