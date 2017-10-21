import sys, glob, csv, codecs
from textblob.classifiers import NaiveBayesClassifier
import nltk
from sklearn.model_selection import train_test_split # requires numpy, scipy
from random import randrange
from tabulate import tabulate


def main():
    data_filenames = sysargs()
    model = create_NB_model(data_filenames)

def division(n, d):
    return round(n / d,2) if d else 0

def create_NB_model(filenames):
    data = extract_data(filenames)
    allData = []
    print("\nConfusion Matrix Style: ")
    print(tabulate([['SPAM', "TP", "FP"], ['HAM', "FN", "TN"]],headers=["",'SPAM', 'HAM'])+"\n")
    for x in range(1, 2):
        # print("Iteration #"+str(x))
        for index, value in enumerate(data):
            labeled_comments = labeledArray(value)
            allData += labeled_comments
            train, test = train_test_split(labeled_comments, shuffle=True, test_size=0.3, random_state=randrange(100))
            classifier = nltk.NaiveBayesClassifier.train(train)
            # print("Size of train: "+ str(len(train))+ " size of test: "+str(len(test)))
            print("Naive Bayes Model Trained On: {}".format(fileandIndex[index]))
            print("Train Size: {} Test Size: {}".format(str(len(train)),str(len(test))))
            # print("Test on {} records from same file".format(str(len(test))))
            truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(classifier, test)
            print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))
            fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
            accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
            truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
            truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
            precision = str(division(truepositives,(truepositives+falsepositives)))
            print("True Positive Rate: {} True Negative Rate: {}, Precision: {}, Accuracy: {}, F-Score: {}".format(truepositiverate,truenegativerate,precision,accuracy,fscore))
            # print("{}|{}\n{}|{}".format(truepositives, truenegatives, falsepositives, falsenegatives ))
            fileList = list(range(0, len(data)))
            fileList.remove(index)
            for z in fileList:
                labeled_comments = labeledArray(data[z])
                truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(classifier, labeled_comments)
                print("\nTesting on {} which has {} records.\n".format(fileandIndex[z], len(labeled_comments)))
                print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))
                fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
                accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
                truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
                truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
                precision = str(division(truepositives,(truepositives+falsepositives)))
                print("True Positive Rate: {} True Negative Rate: {}, Precision: {}, Accuracy: {}, F-Score: {}".format(truepositiverate,truenegativerate,precision,accuracy,fscore))
                # print("Accuracy of model trained on {} and tested on {}: {}".format(fileandIndex[index], fileandIndex[z],str(nltk.classify.accuracy(classifier, labeled_comments))))
            print("\n---------------------------------------------------------------------------------------------------------------\n")
        train, test = train_test_split(allData, shuffle=True, test_size=0.3, random_state=randrange(100))
        classifier = nltk.NaiveBayesClassifier.train(train)
        print("Naive Bayes Model Trained On and Tested on All Data")
        print("Size of train: "+ str(len(train))+ " size of test: "+str(len(test)))
        # print("Accuracy on same file testset: "+str(nltk.classify.accuracy(classifier, test)))
        truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(classifier, test)
        print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))


def testClassifier(classifier, testset):
    truepositives = 0
    truenegatives = 0
    falsepositives = 0
    falsenegatives = 0
    for y in testset:
        guess = classifier.classify({"author": y[0]['author'], "date":  y[0]['date'], "content":  y[0]['content']})
        if y[1] == '0':
            if guess == '0':
                truenegatives = truenegatives+ 1
            elif guess == '1':
                falsepositives = falsepositives+ 1
        elif y[1] == '1':
            if guess == '0':
                falsenegatives = falsenegatives+ 1
            elif guess == '1':
                truepositives = truepositives+ 1
    return truepositives, truenegatives, falsepositives, falsenegatives

def labeledArray(data):
    return ([({"author":comment['AUTHOR'], "date":comment['DATE'],'content': comment['CONTENT']}, comment['CLASS'])for comment in data])



def main_classifier(data):
    training = []
    testing = []
    for x in data:
        for z in x:
            pass


# Returns an array of tuples: [('text here 123 testing', true)]
# The first tuple entry is a comment's text, the second classifies it as SPAM or HAM
# Data file structure:    COMMENT_ID,    AUTHOR,        DATE,        CONTENT,    CLASS
#                         0            1            2            3            4
fileandIndex = []
def extract_data(filenames):
    alldata = []
    for x in filenames:
        print("Loading data for "+x)
        with open(x, 'r') as f:
            thisfilesdata = []
            csvRead = csv.reader(f, delimiter=',')
            namesofcolumns = []
            for index, value in enumerate(csvRead):
                if index == 0:
                    for z in value:
                        namesofcolumns.append(z)
                else:
                    obj = {}
                    for indexTwo, valueTwo in enumerate(value):
                        obj[namesofcolumns[indexTwo]] = valueTwo
                        thisfilesdata.append(obj)
        fileandIndex.append(x)
        alldata.append(thisfilesdata)
    # alldata is an array containing arrays for each filename inputed.
    # inside a files data array (alldata[x])  there is one record
    # A record is a dictionary with these attributes
    # COMMENT_ID AUTHOR DATE CONTENT CLASS
    return alldata

# Returns a tuple: (model_filename, deploy_filename)
def sysargs():
    if (len(sys.argv) > 2):
        datafilenames = sys.argv[1:len(sys.argv)]
        if datafilenames != []:
            return datafilenames
        else:
            print('Error: search strings must match at least one file each.')
            exit(1)
    else:
        print('Usage: python model_on_and_deploy_to.py dataDirectory/*')
        print('Note: search patterns such as ./*.csv can be used to supply multiple files for each set.')
        exit(1)

if (__name__ == '__main__'):
    main()
