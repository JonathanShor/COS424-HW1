
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
import email

from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import metrics
from sklearn.decomposition import PCA

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

def storePreds(path, yHats, paras, start_time):
    outfile= open(path+"predictions_"+str(paras)+"_csec="+str(int(time.time()-start_time))+".txt", 'w')
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
    
def print_top10(clf):
    features_list = read_txt_dat(path+'Train/train_emails_vocab_200.txt','r+')
    features = numpy.asarray(features_list)
    sorted_coef_indices = sorted(range(len(clf.coef_[0])),key=clf.coef_[0].__getitem__)
    print features[sorted_coef_ind[-10:]]  
    
def make_roc_plot(clf, testBow, testClasses):    
    mult_prob = clf.predict_proba(testBow)
    y_score = mult_prob[:,1]
    pos_label = 'Spam'
    fpr,tpr,thresholds = metrics.roc_curve(testClasses,y_score,pos_label)
    roc_auc = metrics.auc(fpr, tpr)

    import pylab as pl  
    pl.clf()
    pl.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC Curve Multinomial Naive Bayes')
    pl.legend(loc = "lower right")
    pl.show()    

def feature_selection(bagofwords,bagofwords_test)
    sel = VarianceThreshold(threshold=(0.05))
    bow_fs=sel.fit_transform(bagofwords)
    bow_fs_test=bagofwords_test[:,sel.get_support()]
    clf.fit(bow_fs,labels)
    bow_fs_test=bagofwords_test[:,sel.get_support()]
    

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
    make_roc = False
    predsf = ''
    start_time = time.time()

    try:
        opts, args = getopt.getopt(argv,"p:o:1:K:bBsSDm:ra:",["path=","ofile="])
    except getopt.GetoptError:
        print 'python text_process.py -p <path> -o <outputfile> -v <vocabulary>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'text_process.py -p <path> -o <outputfile> -v <vocabulary>'
            sys.exit()
        # -a <.txt preds file on which to report accuracy>
        elif opt in "-a":
            predsf = arg
        elif opt in "-r":
            make_roc = True
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
    if predsf:
        print "%s%% accuracy for %s." % (100*get_acc(read_txt_dat(predsf),testC), predsf)
    if alph > -1:
        startclassif_time = time.time()
        clf = MultinomialNB(alpha=alph)
        print clf.fit(trainB,trainC)
        yHats = getPredictions(clf, testB)
        print "Naive Bayes with alpha=%s produced accuracy of %s%%." % (alph, \
                            100*get_acc(yHats,testC))
        storePreds(path, yHats, "numfeatures=%s_NaiveBayes_alpha=%s" % (numfeatures,alph), startclassif_time)

    if K > -1:
        if K>0:
            startclassif_time = time.time()
            clf = neighbors.KNeighborsClassifier(K,'distance')
            print clf.fit(trainB,trainC)
            yHats = getPredictions(clf, testB)
            print "KNN with K=%s produced accuracy of %s%%." % (K, \
                                            100*get_acc(yHats,testC))
            storePreds(path, yHats, "numfeatures=%s_KNN_K=%s" % (numfeatures,K), startclassif_time)
        
# 99.42% K=1, 99.44% K=3, 99.36% K=15
        else:
            for K in range(1,15):
                startclassif_time = time.time()
                clf = neighbors.KNeighborsClassifier(K,'distance')
                print clf.fit(trainB,trainC)
                yHats = getPredictions(clf, testB)
                print "KNN with K=%s produced accuracy of %s%%." % (K, \
                                            100*get_acc(yHats,testC))
                storePreds(path, yHats, "numfeatures=%s_KNN_K=%s" % (numfeatures,K), startclassif_time)


    if SVC:
        startclassif_time = time.time()
        kernel = "linear"
        clf = svm.SVC(kernel = kernel, probability = True)
        print clf.fit(trainB,trainC)
        print "Using %s support vectors, with %s Not Spam and %s Spam vectors." % \
                (len(clf.support_vectors_), clf.n_support_[0], clf.n_support_[1])
        yHats = getPredictions(clf, testB)
        print "SVC with %s kernel produced accuracy of %s%%." % (kernel, 100*get_acc(yHats,testC))
        storePreds(path, yHats, "numfeatures=%s_SVC_kernel=%s" % (numfeatures,kernel), startclassif_time)
        if make_roc:
            make_roc_plot(clf, testB, testC)
    
#        rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(trainC, 2), scoring='accuracy')
#        rfecv.fit(trainB, trainC)
#        print("Optimal number of features for this SVC : %d" % rfecv.n_features_)
# 99.28% with all default para

    if DTree:
        startclassif_time = time.time()
        clf = tree.DecisionTreeClassifier()
        print clf.fit(trainB,trainC)
        yHats = getPredictions(clf, testB)
        print "Decision Tree produced accuracy of %s%%." % (100*get_acc(yHats,testC))
        storePreds(path, yHats, "numfeatures=%s_Dtree" % (numfeatures), startclassif_time)

        from sklearn.externals.six import StringIO
        import pydot
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(path+"DTree.pdf")



    print str(time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])



