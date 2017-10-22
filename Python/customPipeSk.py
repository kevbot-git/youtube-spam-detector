from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from model_on_and_deploy_to import extract_data, sysargs
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from random import randrange
import numpy as np
from tabulate import tabulate

def division(n, d):
    return round(n / d,2) if d else 0

def computeStatistics(truepositives, truenegatives, falsepositives, falsenegatives):
    fscore = str(division((2*truepositives),(2*truepositives+falsepositives+falsenegatives)))
    accuracy = str(division((truepositives+truenegatives),(truepositives+truenegatives+falsepositives+falsenegatives)))
    truepositiverate = str(division(truepositives,(truepositives+falsenegatives)))
    truenegativerate = str(division(truenegatives, (truenegatives+falsepositives)))
    precision = str(division(truepositives,(truepositives+falsepositives)))
    return fscore, accuracy,truepositiverate,truenegativerate, precision

def buildconfusionmetrics(guesses, realities):
    if len(guesses) != len(realities):
        return
    truepositives, truenegatives, falsepositives, falsenegatives = 0, 0 ,0 ,0
    for index, guess in enumerate(guesses):
        if guess == '1':
            if guess == realities[index]:
                truepositives+=1
            else:
                falsepositives+=1
        elif guess == '0':
            if guess == realities[index]:
                truenegatives+=1
            else:
                falsenegatives+=1
    return truepositives, truenegatives, falsepositives, falsenegatives




def main():
    data_filenames = sysargs()
    data = extract_data(data_filenames)
    for y,z in enumerate(data):
        text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('svm',svm.LinearSVC()), ])
        text_clf_bern = Pipeline([('vect', CountVectorizer()), ('classifier', BernoulliNB()),])
        text_clf = Pipeline([('vectorizer', CountVectorizer()),('transformer', TfidfTransformer()),('classifier', MultinomialNB())])
        train, test = train_test_split(z, shuffle=True, test_size=0.3, random_state=randrange(100))
        print("Building a MultinominalNB Classifier, SVM Classifier and a BernoulliNB Classifier with {} records from {}".format(str(len(train)), data_filenames[y]))
        trainlist=[str(x['AUTHOR'])+str(x['DATE'])+str(x['CONTENT'])  for x in train]
        trainlistlabels=[x['CLASS'] for x in train]
        text_clf.fit(trainlist, trainlistlabels)
        text_clf_bern.fit(trainlist, trainlistlabels)
        text_clf_svm.fit(trainlist,trainlistlabels)
        predicted = text_clf.predict([str(x['AUTHOR'])+str(x['DATE'])+str(x['CONTENT']) for x in test])
        predictedsvm = text_clf_svm.predict([str(x['AUTHOR'])+str(x['DATE'])+str(x['CONTENT']) for x in test])
        predictedbern = text_clf_bern.predict([str(x['AUTHOR'])+str(x['DATE'])+str(x['CONTENT']) for x in test])
        testlabels=[x['CLASS'] for x in test]
        truepositives, truenegatives, falsepositives, falsenegatives = buildconfusionmetrics(predicted, testlabels)
        truepositivesvm, truenegativesvm, falsepositivesvm, falsenegativesvm = buildconfusionmetrics(predictedsvm,testlabels)
        truepositivebern, truenegativebern, falsepositivebern, falsenegativebern = buildconfusionmetrics(predictedbern,testlabels)
        print("(MultinominalNB Classifier, SVM Classifier, BernoulliNB Classifier) results on {} records from {} test set".format(str(len(test)), data_filenames[y]))
        print("_______________________")
        print(tabulate([['SPAM', "("+str(truepositives)+", "+str(truepositivesvm)+", "+str(truepositivebern)+")", "("+str(falsepositives)+", "+str(falsepositivesvm)+", "+str(falsepositivebern)+")"], ['HAM', "("+str(falsenegatives)+", "+str(falsenegativesvm)+", "+str(falsenegativebern)+")", "("+str(truenegatives)+", "+str(truenegativesvm)+", "+str(truenegativebern)+")"]],headers=["",'SPAM', 'HAM']))

        print("_______________________")
        for ytoo,ztoo in enumerate(data):
            if (ytoo != y):
                train, test = train_test_split(ztoo, shuffle=True, test_size=0.3, random_state=randrange(100))
                predicted = text_clf.predict([str(x['AUTHOR'])+str(x['DATE'])+str(x['CONTENT']) for x in test])
                predictedsvm = text_clf_svm.predict([str(x['AUTHOR'])+str(x['DATE'])+str(x['CONTENT']) for x in test])
                predictedbern = text_clf_bern.predict([str(x['AUTHOR'])+str(x['DATE'])+str(x['CONTENT']) for x in test])
                testlabels=[x['CLASS'] for x in test]
                truepositives, truenegatives, falsepositives, falsenegatives = buildconfusionmetrics(predicted, testlabels)
                truepositivesvm, truenegativesvm, falsepositivesvm, falsenegativesvm = buildconfusionmetrics(predictedsvm,testlabels)
                truepositivebern, truenegativebern, falsepositivebern, falsenegativebern = buildconfusionmetrics(predictedbern,testlabels)
                fscore, accuracy,truepositiverate,truenegativerate, precision = computeStatistics(truepositives, truenegatives, falsepositives, falsenegatives)
                fscorebern, accuracybern,truepositiveratebern,truenegativeratebern, precisionbern = computeStatistics(truepositivebern, truenegativebern, falsepositivebern, falsenegativebern)
                fscoresvm, accuracysvm,truepositiveratesvm,truenegativeratesvm, precisionsvm = computeStatistics(truepositivesvm, truenegativesvm, falsepositivesvm, falsenegativesvm)
                print("(MultinominalNB Classifier, SVM Classifier, BernoulliNB Classifier) results on {} records from {} test set".format(str(len(test)), data_filenames[ytoo]))
                print("+++++++++++++++++++++++")
                print(tabulate([['SPAM', "("+str(truepositives)+", "+str(truepositivesvm)+", "+str(truepositivebern)+")", "("+str(falsepositives)+", "+str(falsepositivesvm)+", "+str(falsepositivebern)+")"], ['HAM', "("+str(falsenegatives)+", "+str(falsenegativesvm)+", "+str(falsenegativebern)+")", "("+str(truenegatives)+", "+str(truenegativesvm)+", "+str(truenegativebern)+")"]],headers=["",'SPAM', 'HAM']))
                print("\nFScore: ({},{},{}), Accuracy: ({},{},{}), Precision: ({},{},{}), TPR: ({},{},{}), TNR: ({},{},{})".format(fscore,fscoresvm,fscorebern,accuracy,accuracysvm,accuracybern, precision, precisionsvm, precisionbern, truepositiverate, truepositiveratesvm, truepositiveratebern, truenegativerate, truenegativeratesvm, truenegativeratebern))
                print("+++++++++++++++++++++++")
        print("--------------------------------------------------------------------------------------------------------------------------------------\n")
    counter = 0
    while counter < 5:
        trainset = []
        trainsetclasses = []
        testset = []
        testsetclasses= []
        for y,z in enumerate(data):
            if y == counter:
                for record in z:
                    testset.append(str(record['AUTHOR'])+str(record['DATE'])+str(record['CONTENT']))
                    testsetclasses.append(str(record['CLASS']))
            else:
                for record in z:
                    trainset.append(str(record['AUTHOR'])+str(record['DATE'])+str(record['CONTENT']))
                    trainsetclasses.append(str(record['CLASS']))
        text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('svm',svm.LinearSVC()), ])
        text_clf_bern = Pipeline([('vect', CountVectorizer()), ('classifier', BernoulliNB()),])
        text_clf = Pipeline([('vectorizer', CountVectorizer()),('transformer', TfidfTransformer()),('classifier', MultinomialNB())])
        text_clf.fit(trainset, trainsetclasses)
        text_clf_svm.fit(trainset, trainsetclasses)
        text_clf_bern.fit(trainset, trainsetclasses)
        predict = text_clf.predict(testset)
        predictedsvm = text_clf_svm.predict(testset)
        predictedbern = text_clf_bern.predict(testset)
        truepositivesvm, truenegativesvm, falsepositivesvm, falsenegativesvm = buildconfusionmetrics(predictedsvm, testsetclasses)
        truepositives, truenegatives, falsepositives, falsenegatives = buildconfusionmetrics(predict, testsetclasses)
        truepositivebern, truenegativebern, falsepositivebern, falsenegativebern = buildconfusionmetrics(predictedbern,testsetclasses)
        fscore, accuracy,truepositiverate,truenegativerate, precision = computeStatistics(truepositives, truenegatives, falsepositives, falsenegatives)
        fscorebern, accuracybern,truepositiveratebern,truenegativeratebern, precisionbern = computeStatistics(truepositivebern, truenegativebern, falsepositivebern, falsenegativebern)
        fscoresvm, accuracysvm,truepositiveratesvm,truenegativeratesvm, precisionsvm = computeStatistics(truepositivesvm, truenegativesvm, falsepositivesvm, falsenegativesvm)
        print("Building a MultinominalNB Classifier, SVM Classifier and a BernoulliNB Classifier with {} records from all files except {}".format(str(len(trainset)), data_filenames[counter]))
        print("(NaiveBayesClassifier, SVM Classifier BernoulliNB Classifier) results on {} records from {} test set".format(str(len(testset)), data_filenames[counter]))
        print(tabulate([['SPAM', "("+str(truepositives)+", "+str(truepositivesvm)+", "+str(truepositivebern)+")", "("+str(falsepositives)+", "+str(falsepositivesvm)+", "+str(falsepositivebern)+")"], ['HAM', "("+str(falsenegatives)+", "+str(falsenegativesvm)+", "+str(falsenegativebern)+")", "("+str(truenegatives)+", "+str(truenegativesvm)+", "+str(truenegativebern)+")"],['ACTUAL',0,0]],headers=["",'SPAM', 'HAM']))
        print("FScore: ({},{},{}), Accuracy: ({},{},{}), Precision: ({},{},{}), TPR: ({},{},{}), TNR: ({},{},{})".format(fscore,fscoresvm,fscorebern,accuracy,accuracysvm,accuracybern, precision, precisionsvm, precisionbern, truepositiverate, truepositiveratesvm, truepositiveratebern, truenegativerate, truenegativeratesvm, truenegativeratebern))
        print("--------------------------------------------------------------------------------------------------------------------------------------\n")
        counter+=1





if (__name__ == '__main__'):
    main()
