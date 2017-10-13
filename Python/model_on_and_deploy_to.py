import sys, glob, csv, codecs
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split # requires numpy, scipy
from random import randrange

def main():
    data_filenames = sysargs()
    model = create_model(data_filenames)

def create_model(filenames):
    data = extract_data(filenames)
    for x in range(1, 100):
        print("Iteration #"+str(x))
        for z in data:
            train, test = train_test_split(z, shuffle=True, test_size=0.3, random_state=randrange(100))
            print("Size of train: "+ str(len(train))+ " size of test: "+str(len(test)))
    classifier = []

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
        alldata.append(thisfilesdata)
    # alldata is an array containing arrays for each filename inputed.
    # inside a files data array (alldata[x])  there is one record
    # A record is a dictionary with these attributes
    # COMMENT_ID AUTHOR DATE CONTENT CLASS
    return alldata

# Returns a tuple: (model_filename, deploy_filename)
def sysargs():
    if (len(sys.argv) > 2):
        datafilenames = sys.argv[1:len(sys.argv)-1]
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
