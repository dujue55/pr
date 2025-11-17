'''
This script contains the skeleton code for the sample assignment-1  document classification.
using the kNN classifier. Here you will be using the kNN classifier implemented in the knn.py file
and get hands dirty with a real world dataset
For the optional part of the assignment you can use the sklearn implementation of the tf-idf vectoriser.
'''
import re
from typing import Tuple
import numpy as np
import pandas as pd
from knn import KNNClassifier
import os


class DocumentPreprocessing:
    '''
    Class to process the text data and convert it to a bag of words model
    '''
    def __init__(self,path):
        self.path = path
        self.data = self.load_data()
        self.domain, self.abstract=  self.extract_labels_and_text(self.data)
        self.class_labels = None
        self.generate_labels()
        self.y_train = self.preprocess_labels(self.domain)
        self.X_train = None
        self.vocabulory = None
    def load_data(self):
        '''    
        Extract the data from the csv file and return the data as a pandas dataframe
        '''
    
        # Load the data from the csv file
        #  Extract the classes present in the domain column
        # Complete your code here
        return pd.read_csv(self.path)

    def extract_labels_and_text(self,data : pd.DataFrame()):
        '''
        Extract the classes in the dataset and the text data
        and save them in the domain and abstract variables respectively
        The outputs are the list of classes and the list of text data
        '''
        # Complete your code here
        domain = data["Domain"].tolist()
        abstract = data["Abstract"].tolist()


        return (domain, abstract)
    
    def generate_labels(self):
        '''
        Use the domain variable to generate the class labels and 
        save them in the self.class_labels variable
        Example : if self.domain = ['ab', 'cds','aab', 'aab', 'ab', 'cds']
        then self.class_labels = ['ab', 'aab', 'cds']
        '''
        # Complete your code here

        self.class_labels = np.unique(self.domain).tolist() # Change this line to the correct code

    def preprocess_labels(self, y_train : list()) -> list():
        '''
        From the text based class labels, convert them to integers
        using the labels generated in the generate_labels function
        Examples : if self.domain = ['ab', 'cds','aab', 'aab', 'ab', 'cds']
        then the out put is  [0, 2, 1, 1, 0, 2]
        '''
        # Complete your code here
        label2id = {label: idx for idx, label in enumerate(self.class_labels)}
        
        return  [label2id[label] for label in y_train]

    def remove_special_characters(self,word):
        '''
        This function removes the special characters from the word and returns the cleaned word
        '''
        pattern = r'[^a-zA-Z0-9 ]'  # This pattern only keeps the alphanumeric characters
        # Use the re.sub() function to replace matches with an empty string
        cleaned_word = re.sub(pattern, ' ', word)
        return cleaned_word

    def preprocess(self,text: str ) -> list:
        '''
        Function to preprocess the raw text data
        1. Use the function remove_special_characters to remove the special characters
        2. Remove the words of length 1
        3. Convert to lower case
        return the preprocessed text as a list of words
        '''
        # Complete your code here
        cleaned = self.remove_special_characters(text)
        cleaned = cleaned.lower()
        words = cleaned.split()
        words = [w for w in words if len(w) > 1]
        return words
    
    def bag_words(self):
        '''
        Function to convert the text data to a bag of words model.
         
        will break the task into smaller parts below to make it easier to
        understand the process
        '''
        vocabulory = []
        # Get the unique words in the dataset and sort them in alphabetical order
        # Complete your code here
        for text in self.abstract:
            words = self.preprocess(text)
            vocabulory.extend(words)

        vocabulory = sorted(list(set(vocabulory)))
        word2id = {w:idx for idx, w in enumerate(vocabulory)}
        self.vocabulory = vocabulory

        # Conver the text to a bag of words model
        # Note: the vector contains the count of the words in the text
        X_train = np.zeros((len(self.abstract), len(vocabulory)))

        # Complete your code here
        # Hint: use the preprocess function to preprocess the text data
        for i, text in enumerate(self.abstract):
            words = self.preprocess(text)
            for w in words:
                if w in word2id:
                    j = word2id[w]
                    X_train[i][j] +=1
      
        self.X_train = X_train

    def transform(self,text: list() ) -> np.array:
        '''
        The function takes a list of text data and outputs the 
        feature matrix for the text data.
        Examples if the text is ['this is a test', 'this is another test']
        The output is a numpy array of shape (2, len(self.vocabulory))
        '''
        
        # Complete your code here
        text_matrix = np.zeros((len(text),len(self.vocabulory)))
        word2id = {w: idx for idx, w in enumerate(self.vocabulory)}

        for i,doc in enumerate(text):
            words = self.preprocess(doc)
            for w in words:
                if w in word2id:
                    j = word2id[w]
                    text_matrix[i][j] += 1


        return text_matrix





if __name__ == '__main__':
   
    # Make sure to change the path to appropriate location where the data is stored

    
    trainpath  = './data/webofScience_train.csv'
    testPath = './data/webofScience_test.csv'


    # Create an object of the class document_preprocessing
    document = DocumentPreprocessing(trainpath)
    document.load_data()
    document.bag_words()

    # Some test cases to check if the implementation is correct or not 
    # Note: these case only work for the webofScience dataset provided 
    # You can comment this out this section when you are done with the implementation
    if(document.vocabulory[10] == '0026'):
        print('Test case 1 passed')
    else:
        print('Test case 1 failed')

    if(document.vocabulory[100] == '135'):
        print('Test case 2 passed')
    else:
        print('Test case 2 failed')

    if(document.vocabulory[1000] == 'altitude'):
        print('Test case 3 passed')
    else:
        print('Test case 3 failed')


    # First 10 words in the vocabulory are:
    #['000', '00005', '0001', '0002', '0004', '0005', '0007', '001', '0016', '002']

    print(document.vocabulory[:10])

    pd_Test = pd.read_csv(testPath)

    domain_test, abstract_test = document.extract_labels_and_text(pd_Test)
    y_test = document.preprocess_labels(domain_test)
    X_test = document.transform(abstract_test)

    # Create a kNN classifier object
    knn = KNNClassifier(k=3)

    # Train the kNN classifier
    knn.train(document.X_train, np.array(document.y_train))

    # Compute accuracy on the test set
    accuracy = knn.compute_accuracy(X_test, np.array(y_test))

    # Print the accuracy should be greater than 0.3
    print('Accuracy of the classifier is ', accuracy)

    # For the optional part of the assignment
    # You can use the sklearn implementation of the tf-idf vectoriser
    # The documentation can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    
    







    
