import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def getClassifierObj(classifier, rs):
    '''Functio to get the classifier object
    '''
    if classifier == 'XGBoost':
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
    
    elif classifier == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = rs)
    
    elif classifier == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    
    elif classifier == 'SVM':
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = rs)
        
    elif classifier == 'Kernel SVM':
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = rs)
        
    elif classifier == 'NB':
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        
    elif classifier == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = rs)
        
    elif classifier == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = rs)
        
    return classifier
            
def fitAndPredict(estimator,X_train,y_train,X_test):
    '''Function to fit and predict
    '''
    estimator.fit(X_train,y_train)
    predictions = estimator.predict(X_test)
    return predictions

def getModelAccuracy(y_test, y_pred):
    '''Function to get model accuracy'''
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)*100

def getFinalPredictions(X_train,X_test, y_train, y_test):
    ''' Function to get classifier with highest accuracy on the data
    '''
    classifiers = ['LogisticRegression','KNN', 'NB', 'Kernel SVM','DecisionTree','RandomForest','XGBoost']
    accuracy = 0
    best_predictions = ''
    bestClassifierName = ''
    
    for classifierName in classifiers:
        print('Evaluation started for ', classifierName)
        classifier = getClassifierObj(classifierName,1)
        y_pred = fitAndPredict(classifier,X_train,y_train,X_test)
        
        classifierAccuracy = getModelAccuracy(y_test,y_pred)
        print('Average accuracy of {} is {:.2f}%'.format(classifierName,classifierAccuracy))
        if classifierAccuracy > accuracy :
            accuracy = classifierAccuracy
            bestClassifierName = classifierName
            best_predictions = y_pred
            
            
    print('Classifier with highest accuracy is {}'.format(bestClassifierName))
    return best_predictions, accuracy       

def classify():
    #Get Data
    main_data = pd.read_csv('I:\\Learnings\\Machine Learning\\Projects\\Kaggle\\Forest Cover Type\\train.csv')
    # main_data = pd.read_csv('I:\\Learnings\\Machine Learning\\Projects\\Kaggle\\Forest Cover Type\\train.csv')

    #select features for model building
    X = main_data.iloc[:,1:-1].values
    y = main_data.iloc[:,-1].values

    #split into Training and Test set
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size = 0.2)

    final_pred, final_accuracy = getFinalPredictions(X_train,X_test, y_train, y_test)    
    
    cm = confusion_matrix(y_test, final_pred)
    print(cm)

classify()
