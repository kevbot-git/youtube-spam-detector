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

def train_and_test_naive_bayes(train, test, filenam):
    train = labeledArray(train)
    test = labeledArray(test)
    truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(nltk.NaiveBayesClassifier.train(train), test)
    fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
    accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
    truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
    truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
    precision = str(division(truepositives,(truepositives+falsepositives)))
    print("\nNaive Bayes model trained on all files apart from {} which is the testing file.".format(filenam))
    print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))
    print("True Positive Rate: {} True Negative Rate: {}, Precision: {}, Accuracy: {}, F-Score: {}".format(truepositiverate,truenegativerate,precision,accuracy,fscore))

# def train_and_test_nb(index, value):
#     labeled_comments = labeledArray(value)
#     train, test = train_test_split(labeled_comments, shuffle=True, test_size=0.3, random_state=randrange(100))
#     classifier = nltk.NaiveBayesClassifier.train(train)
#     print("Naive Bayes Model Trained On: {}".format(fileandIndex[index]))
#     print("Train Size: {} Test Size: {}".format(str(len(train)),str(len(test))))
#     truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(classifier, test)
#     print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))
#     fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
#     accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
#     truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
#     truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
#     precision = str(division(truepositives,(truepositives+falsepositives)))
#     print("True Positive Rate: {} True Negative Rate: {}, Precision: {}, Accuracy: {}, F-Score: {}".format(truepositiverate,truenegativerate,precision,accuracy,fscore))
#     return classifier, labeled_comments

def create_NB_model(filenames):
    data = extract_data(filenames)
    labeleddata = []
    print("\nConfusion Matrix Style: ")
    print(tabulate([['SPAM', "TP", "FP"], ['HAM', "FN", "TN"]],headers=["",'SPAM', 'HAM'])+"\n")
    for index, value in enumerate(data):
        labeleddata.append(labeledArray(value))
    train_test = {'train': [], 'test': []}
    for x in labeleddata:
        train, test = train_test_split(x, shuffle=True, test_size=0.3, random_state=randrange(100))
        train_test['train'].append(train)
        train_test['test'].append(test)
    print("Naive Bayes Models")
    for index, train in enumerate(train_test['train']):
        classifier = nltk.NaiveBayesClassifier.train(train)
        print(fileandIndex[index]+" model\n")
        print("Size of training set: "+str(len(train)))
        for testindex, test in enumerate(train_test['test']):
            truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(classifier, test)
            print("\n-----------------------------------------------------\n")
            print("Result on "+fileandIndex[testindex]+" test set")
            print("Size of test set: "+str(len(test)))
            print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))
            fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
            accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
            truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
            truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
            precision = str(division(truepositives,(truepositives+falsepositives)))
            print("True Positive Rate: {} True Negative Rate: {}, Precision: {}, Accuracy: {}, F-Score: {}".format(truepositiverate,truenegativerate,precision,accuracy,fscore))
            print("\n---------------------------------------------------------------------------------------------------------------\n")
        print("\n---------------------------------------------------------------------------------------------------------------\n")

    for index, test in enumerate(labeleddata):
        training_on_these = list(range(0,5))
        training_on_these.remove(index)
        labeled = []
        for z in training_on_these:
            labeled +=labeleddata[z]
        print("Inside loop with "+str(type(labeled))+" of size: "+str(len(labeled)))
        print("Training a model excluding: "+fileandIndex[index])
        classifier = nltk.NaiveBayesClassifier.train(labeled)
        truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(classifier, labeleddata[index])
        print("\n-----------------------------------------------------\n")
        print("Result on "+fileandIndex[index]+" test set "+str(len(labeleddata[index])))
        print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))
        fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
        accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
        truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
        truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
        precision = str(division(truepositives,(truepositives+falsepositives)))
        print("True Positive Rate: {} True Negative Rate: {}, Precision: {}, Accuracy: {}, F-Score: {}".format(truepositiverate,truenegativerate,precision,accuracy,fscore))
        print("\n---------------------------------------------------------------------------------------------------------------\n")







        # classifier, labeled_data = train_and_test_nb(index, value)
        # allData += labeled_data
        # fileList = list(range(0, len(data)))
        # fileList.remove(index)
        # for z in fileList:
        #     labeled_comments = labeledArray(data[z])
        #     print("\nTesting on {} which has {} records.\n".format(fileandIndex[z], len(labeled_comments)))
        #     print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))
        #     fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
        #     accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
        #     truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
        #     truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
        #     precision = str(division(truepositives,(truepositives+falsepositives)))
        #     print("True Positive Rate: {} True Negative Rate: {}, Precision: {}, Accuracy: {}, F-Score: {}".format(truepositiverate,truenegativerate,precision,accuracy,fscore))
        # print("\n---------------------------------------------------------------------------------------------------------------\n")
#     train_and_test_naive_bayes(data[0]+data[1]+data[2]+data[3], data[4], fileandIndex[4])
#     train_and_test_naive_bayes(data[0]+data[1]+data[2]+data[4], data[3], fileandIndex[3])
#     train_and_test_naive_bayes(data[0]+data[1]+data[3]+data[4], data[2], fileandIndex[2])
#     train_and_test_naive_bayes(data[0]+data[2]+data[3]+data[4], data[1], fileandIndex[1])
#     train_and_test_naive_bayes(data[1]+data[2]+data[3]+data[4], data[0], fileandIndex[0])
#     train, test = train_test_split(allData, shuffle=True, test_size=0.3, random_state=randrange(100))
#     classifier = nltk.NaiveBayesClassifier.train(train)
#     print("\nNaive Bayes Model Trained On and Tested on All Data")
#     print("Size of train: "+ str(len(train))+ " size of test: "+str(len(test)))
#     # print("Accuracy on same file testset: "+str(nltk.classify.accuracy(classifier, test)))
#     truepositives, truenegatives, falsepositives, falsenegatives = testClassifier(classifier, test)
#     print(tabulate([['SPAM', truepositives, falsepositives], ['HAM', falsenegatives, truenegatives]],headers=["",'SPAM', 'HAM']))


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
