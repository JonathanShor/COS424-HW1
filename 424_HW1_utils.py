
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time

from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV

NUM_TRAIN_EMAILS = 45000
NUM_TEST_EMAILS = 5000
chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def get_files(mypath):
    return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile, numofemails=NUM_TRAIN_EMAILS):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def print_one_vector(myfile, vectornum=0, numofemails=NUM_TRAIN_EMAILS):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    print 'Printing feature vector num %s: %s' % (vectornum, bagofwords[vectornum])
    #    print len(bagofwords[vectornum])
    return bagofwords[vectornum]

# return list of .txt file created by this or email_process.py
def read_txt_dat(myfile):
    txtdat = []
    inputf = open(myfile,'r')
    for line in inputf:
        txtdat.append(line.rstrip())
    return txtdat

# path should be to folder containing Test & Train folders with trailing /
# emailcutoff should match the cutoff # in the filenames
def gather_data(path, emailcutoff=200):
    trainBow = read_bagofwords_dat(path+'Train/train_emails_bag_of_words_'+str(emailcutoff)+'.dat',NUM_TRAIN_EMAILS)
    trainClasses = read_txt_dat(path+'Train/train_emails_classes_'+str(emailcutoff)+'.txt')
    testBow = read_bagofwords_dat(path+'Test/test_emails_bag_of_words_0.dat',NUM_TEST_EMAILS)
    testClasses = read_txt_dat(path+'Test/test_emails_classes_0.txt')
    return(trainBow, trainClasses, testBow, testClasses)


def get_acc(yHs, master):
    i=-1
    correct = 0
    for c in master:
        i += 1
        if c == yHs[i]:
            correct +=1
    return float(correct)/len(master)

def doPrediction(clf, testBow, testClasses):
    yHats = clf.predict(testBow)
    return get_acc(yHats,testClasses)

def getPredictions(clf, testBow):
    return clf.predict(testBow)

def storePreds(path, yHats, paras):
    outfile= open(path+"predictions_"+str(paras)+"_"+str(int(time.time()))+".txt", 'w')
    outfile.write("\n".join(yHats))
    outfile.close()

def getMisclassified(yHats, grounds):
    misses = []
    i = 0
    for yh in yHats:
        if yh.lower() != grounds[i][3:grounds[i].rfind("_")].lower():
            misses.append(grounds[i])
        #            print str(i) + ":" + grounds[i]
        i += 1
    return misses

def getCommonMissed(path, numfeatures, grounds):
    files = get_files(path)
    files = [f for f in files if f.find(numfeatures) > -1]
    common = getMisclassified(read_txt_dat(path+files[0]), grounds)
    for f in files[1:]:
        intersect(common, getMisclassified(read_txt_dat(path+f),grounds))
    return common


# path should have one Train and one Test folder, each of which should
# have one folder for each class. Class folders should
# contain text documents that are labeled with the class label (folder
# name).
def main(argv):
    path = ''
    outputf = ''
    vocabf = ''
    printone = -1
    K = -1
    alph = -1.
    SVC = False
    DTree = False
    missedcheck = ""
    start_time = time.time()

    try:
        opts, args = getopt.getopt(argv,"p:o:1:K:bBsSDm:",["path=","ofile=","printvector=","k=","s=","b="])
    except getopt.GetoptError:
        print 'python text_process.py -p <path> -o <outputfile> -v <vocabulary>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'text_process.py -p <path> -o <outputfile> -v <vocabulary>'
            sys.exit()
        elif opt in ("-S","-s"):
            SVC = True
        elif opt in "-m":
            missedcheck = arg
        elif opt in ("-D","-d"):
            DTree = True
        elif opt in ("-K", "--k"):
            K = int(arg)
        elif opt in ("-b", "--b"):
            alph = float(arg)
        elif opt in ("-B"):
            alph = 1.
        elif opt in ("-1", "--printvector"):
            printone = int(arg)
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-o", "--ofile"):
            outputf = arg

    print 'Path is "', path
#    print 'Output file name is "', outputf
#    print 'vocabulary file is "', vocabf
    if printone > -1:
        print_one_vector(path+outputf,5)
#    sys.exit()
    (trainB, trainC, testB, testC) = gather_data(path)
    numfeatures = len(trainB[0])
    if missedcheck:
        comm = getCommonMissed(path, missedcheck, read_txt_dat(path+'Test/test_emails_samples_class_0.txt'))
        print comm
        print "%s emails misclassified among all predictions with %s in common." % (len(comm), missedcheck)
    if alph > -1:
        clf = MultinomialNB(alpha=alph)
        print clf.fit(trainB,trainC)
        yHats = getPredictions(clf, testB)
        print "Naive Bayes with alpha=%s produced accuracy of %s%%." % (alph, \
                            100*get_acc(yHats,testC))
        storePreds(path, yHats, "numfeatures=%s_NaiveBayes_alpha=%s" % (numfeatures,alph))

    if K > -1:
        if K>0:
            clf = neighbors.KNeighborsClassifier(K,'distance')
            print clf.fit(trainB,trainC)
            yHats = getPredictions(clf, testB)
            print "KNN with K=%s produced accuracy of %s%%." % (K, \
                                            100*get_acc(yHats,testC))
            storePreds(path, yHats, "numfeatures=%s_KNN_K=%s" % (numfeatures,K))
        
# 99.42% K=1, 99.44% K=3, 99.36% K=15
        else:
            for K in range(1,15):
                clf = neighbors.KNeighborsClassifier(K,'distance')
                print clf.fit(trainB,trainC)
                yHats = getPredictions(clf, testB)
                print "KNN with K=%s produced accuracy of %s%%." % (K, \
                                            100*get_acc(yHats,testC))
                storePreds(path, yHats, "numfeatures=%s_KNN_K=%s" % (numfeatures,K))


    if SVC:
        kernel = "linear"
        clf = svm.SVC(kernel = kernel)
        print clf.fit(trainB,trainC)
        print "Using %s support vectors, with %s Not Spam and %s Spam vectors." % \
                (len(clf.support_vectors_), clf.n_support_[0], clf.n_support_[1])
        yHats = getPredictions(clf, testB)
        print "SVC with %s kernel produced accuracy of %s%%." % (kernel, 100*get_acc(yHats,testC))
        storePreds(path, yHats, "numfeatures=%s_SVC_kernel=%s" % (numfeatures,kernel))

        
#        rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(trainC, 2), scoring='accuracy')
#        rfecv.fit(trainB, trainC)
#        print("Optimal number of features for this SVC : %d" % rfecv.n_features_)
# 99.28% with all default para

    if DTree:
        clf = tree.DecisionTreeClassifier()
        print clf.fit(trainB,trainC)
        yHats = getPredictions(clf, testB)
        print "Decision Tree produced accuracy of %s%%." % (100*get_acc(yHats,testC))
        storePreds(path, yHats, "numfeatures=%s_Dtree" % (numfeatures))

        from sklearn.externals.six import StringIO
        import pydot
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(path+"DTree.pdf")



    print str(time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])



