from model_on_and_deploy_to import extract_data, sysargs
import nltk
from nltk.corpus import stopwords
import csv
def main():
    data_filenames = sysargs()
    data = extract_data(data_filenames)
    pos = {}
    stopword = {}
    chunking = {}
    for x in data_filenames:
        pos[x] = []
        stopword[x] = []
        chunking[x] = []
    for filenumber, contains in enumerate(data):
        for record in contains:
            posrecord = dict(record)
            stoprecord = dict(record)
            chunkedrecord = dict(record)
            posrecord['CONTENT'] = ' '.join([word + '/' + pos for word, pos in nltk.pos_tag(nltk.word_tokenize(posrecord['CONTENT']))])
            stoprecord['CONTENT'] = ' '.join([word for word in nltk.word_tokenize(stoprecord['CONTENT']) if not word in stopwords.words('english')])
            chunkedrecord['CONTENT'] = nltk.chunk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(chunkedrecord['CONTENT']))).freeze()
            pos[data_filenames[filenumber]].append(posrecord)
            stopword[data_filenames[filenumber]].append(stoprecord)
            chunking[data_filenames[filenumber]].append(chunkedrecord)
    for x in pos.keys():
        with open("PosData/"+x.split('/')[1], 'w') as f:
            write = csv.DictWriter(f, fieldnames=["CONTENT", "DATE", "AUTHOR", "COMMENT_ID", "CLASS"])
            write.writeheader()
            for data in pos[x]:
                write.writerow(data)
    for x in stopword.keys():
        with open("StopData/"+x.split('/')[1], 'w') as f:
            write = csv.DictWriter(f, fieldnames=["CONTENT", "DATE", "AUTHOR", "COMMENT_ID", "CLASS"])
            write.writeheader()
            for data in stopword[x]:
                write.writerow(data)
    for x in chunking.keys():
        with open("ChunkData/"+x.split('/')[1], 'w') as f:
            write = csv.DictWriter(f, fieldnames=["CONTENT", "DATE", "AUTHOR", "COMMENT_ID", "CLASS"])
            write.writeheader()
            for data in chunking[x]:
                write.writerow(data)

if (__name__ == '__main__'):
    main()
