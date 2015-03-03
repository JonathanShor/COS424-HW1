
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

def storePerfStats(path, feats, grounds):
    files = get_files(path)
    files = [f for f in files if f.find(feats) > -1]
    outfile = open(path+"Stats_"+str(feats)[:-1]+".txt",'w')
    outfile.write("Accuracy\tPrecision\tRecall\tF1\t\tFilename\n")
    outf = open(path+"Stats_"+str(feats)[:-1]+"SUM.txt",'w')
    for f in files:
        yHs = read_txt_dat(path+f)
        outfile.write("%s%%\t" % (100*get_acc(yHs,grounds)))
        outfile.write("%s\t\t" % metrics.precision_score([1 if x=='Spam' else 0 for x in grounds], \
                                                       [1 if x=='Spam' else 0 for x in yHs]))
        outfile.write("%s\t" % metrics.recall_score([1 if x=='Spam' else 0 for x in grounds], \
                                                    [1 if x=='Spam' else 0 for x in yHs]))
        outfile.write("%s\t" % metrics.f1_score([1 if x=='Spam' else 0 for x in grounds], \
                                                [1 if x=='Spam' else 0 for x in yHs]))
        outfile.write("%s\n" % f)
        outf.write("%s\n%s\n" % (f,metrics.classification_report(grounds, yHs)))
    outfile.close()
    return path+"Stats_"+str(feats)+".txt"


def getMisclassified(yHats, grounds):
    misses = []
    i = 0
    for yh in yHats:
        if yh.lower() != grounds[i][3:grounds[i].rfind("_")].lower():
            misses.append(grounds[i])
        i += 1
    return misses

def getCommonMissed(path, numfeatures, grounds):
    files = get_files(path)
    files = [f for f in files if f.find(numfeatures) > -1]
    common = getMisclassified(read_txt_dat(path+files[0]), grounds)
    for f in files[1:]:
        intersect(common, getMisclassified(read_txt_dat(path+f),grounds))
    return common
    
def print_top10(clf, path):
    features_list = read_txt_dat(path+'Train/train_emails_vocab_200.txt')
    features = numpy.asarray(features_list)
    sorted_coef_indices = sorted(range(len(clf.coef_[0])),key=clf.coef_[0].__getitem__)
    print features[sorted_coef_indices[-10:]]
    print (sorted(clf.coef_[0]))[-10:]

def print_DTreetop10(clf, path):
    features_list = read_txt_dat(path+'Train/train_emails_vocab_200.txt')
    features = numpy.asarray(features_list)
    sorted_imp_indices = sorted(range(len(clf.feature_importances_)),key=clf.feature_importances_.__getitem__)
    print features[sorted_imp_indices[-10:]]
    print (sorted(clf.feature_importances_))[-10:]
    
def make_roc_plot(clf, testBow, testClasses, clfIs='Multinomial Naive Bayes'):
    if clfIs == 'LSVC':
        mult_prob = clf.decision_function(testBow)
        y_score = mult_prob
    else:
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
    if clfIs == 'LSVC':
        pl.title('ROC Curve Linear SVM')
    elif clfIs == 'DTree':
        pl.title('ROC Curve Decision Tree')
    elif clfIs[0:3] == 'KNN':
        pl.title('ROC Curve %s' % clfIs)
    else:
        pl.title('ROC Curve Multinomial Naive Bayes')
    pl.legend(loc = "lower right")
    pl.show()    

def feature_selection(bagofwords,bagofwords_test):
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
    printone = -1
    K = -1
    alph = -1.
    SVC = False
    DTree = False
    missedcheck = ""
    make_roc = False
    predsf = ''
    feat_select = False
    doStore = True
    getStats = ""
    start_time = time.time()

    try:
        opts, args = getopt.getopt(argv,"p:K:b:m:t:a:BsSDRrX",["path="])
    except getopt.GetoptError:
        print 'python text_process.py -p <path> -K <num_neighbors> -b <alpha> \
                -[mt] <parameters> -a <filename> -[SDRX]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'python text_process.py -p <path> -K <num_neighbors> -b <alpha> \
                    -[mt] <parameters> -a <filename> -[SDRX]'
            sys.exit()
        # -a <.txt preds file on which to report accuracy>
        elif opt in "-a":
            predsf = arg
        # -X will stop any prediction dump files from being created
        elif opt in "-X":
            doStore = False
        elif opt in ("-R","-r"):
            make_roc = True
        elif opt in ("-S","-s"):
            SVC = True
        # -m <string to match in -p path dir> all matched files will be
        # processed for common misclassified emails
        elif opt in "-m":
            missedcheck = arg
        # -t <string to match in -p path dir> all matched files will be
        # processed for performance stats in storePerfStats
        elif opt in "-t":
            getStats = arg
        elif opt in ("-D","-d"):
            DTree = True
        # K = 0 produces KNN for [1-14], exploring the hyperparameter
        # K > 0 produces KNN for that K
        elif opt in ("-K", "--k"):
            K = int(arg)
        elif opt in ("-b", "--b"):
            alph = float(arg)
        elif opt in ("-B"):
            alph = 1.
        elif opt in ("-p", "--path"):
            path = arg

    print 'Path is "', path
    (trainB, trainC, testB, testC) = gather_data(path)
    numfeatures = len(trainB[0])

    if missedcheck:
        comm = getCommonMissed(path, missedcheck, read_txt_dat(path+'Test/test_emails_samples_class_0.txt'))
        print comm
        print "%s emails misclassified among all predictions with %s in common." \
                % (len(comm), missedcheck)
    if predsf:
        print "%s%% accuracy for %s." % (100*get_acc(read_txt_dat(predsf),testC), predsf)
    if getStats:
        print "%s produced with requested stats for %s." \
                % (storePerfStats(path, getStats, testC), getStats)
    if alph > -1:
        startclassif_time = time.time()
        clf = MultinomialNB(alpha=alph)
        print clf.fit(trainB,trainC)
        yHats = getPredictions(clf, testB)
        print "Naive Bayes with alpha=%s produced accuracy of %s%%." % (alph, \
                            100*get_acc(yHats,testC))
        if doStore:
            storePreds(path, yHats, "numfeats=%s_NaiveBayes_alpha=%s" % (numfeatures,alph), \
                       startclassif_time)

    if K > -1:
        if K>0:
            startclassif_time = time.time()
            clf = neighbors.KNeighborsClassifier(K,'distance')
            print clf.fit(trainB,trainC)
            yHats = getPredictions(clf, testB)
            print "KNN with K=%s produced accuracy of %s%%." % (K, \
                                            100*get_acc(yHats,testC))
            if doStore:
                storePreds(path, yHats, "numfeats=%s_KNN_K=%s" % (numfeatures,K), startclassif_time)
            if make_roc:
                make_roc_plot(clf, testB, testC, clfIs='KNN, K=%s' % K)


        else:
            for K in range(1,15):
                startclassif_time = time.time()
                clf = neighbors.KNeighborsClassifier(K,'distance')
                print clf.fit(trainB,trainC)
                yHats = getPredictions(clf, testB)
                print "KNN with K=%s produced accuracy of %s%%." % (K, \
                                            100*get_acc(yHats,testC))
                if doStore:
                    storePreds(path, yHats, "numfeats=%s_KNN_K=%s" % (numfeatures,K), \
                               startclassif_time)


    if SVC:
        startclassif_time = time.time()
        clf = svm.LinearSVC(tol=0.00001)
        print clf.fit(trainB,trainC)
            #        print "Using %s support vectors, with %s Not Spam and %s Spam vectors." % \
            #    (len(clf.support_vectors_), clf.n_support_[0], clf.n_support_[1])
        yHats = getPredictions(clf, testB)
        print "LinearSVC produced accuracy of %s%%." % (100*get_acc(yHats,testC))
        if doStore:
            storePreds(path, yHats, "numfeats=%s_LSVC" % (numfeatures), startclassif_time)
        print_top10(clf, path)
        if make_roc:
            make_roc_plot(clf, testB, testC, clfIs='LSVC')
        if feat_select:
            rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(trainC, 2), scoring='accuracy')
            rfecv.fit(trainB, trainC)
            print("Optimal number of features for this SVC : %d" % rfecv.n_features_)

    if DTree:
        startclassif_time = time.time()
        clf = tree.DecisionTreeClassifier()
        print clf.fit(trainB,trainC)
        yHats = getPredictions(clf, testB)
        print "Decision Tree produced accuracy of %s%%." % (100*get_acc(yHats,testC))
        if doStore:
            storePreds(path, yHats, "numfeats=%s_Dtree" % (numfeatures), startclassif_time)
        print_DTreetop10(clf, path)
        if make_roc:
            make_roc_plot(clf, testB, testC, clfIs='DTree')


#        from sklearn.externals.six import StringIO
#        import pydot
#        dot_data = StringIO()
#        tree.export_graphviz(clf, out_file=dot_data)
#        graph = pydot.graph_from_dot_data(dot_data.getvalue())
#        graph.write_pdf(path+"DTree.pdf")



    print str(time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])



