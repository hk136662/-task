# -task
#word2vec+SVMによる感情分析タスク
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn import svm
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

with open("...................","r") as infile:
    pos_tweets=infile.readlines()
with open("...................","r") as infile:
    neg_tweets=infile.readlines()
    
y=np.concatenate((np.ones(len(pos_tweets)),np.zeros(len(neg_tweets))))

x_train,x_test,y_train,y_test=train_test_split(np.concatenate((pos_tweets,neg_tweets)),y,test_size=0.2)

def cleanText(corpus):
    corpus=[z.lower().replace("/n","").split() for z in corpus]
    return corpus
x_train=cleanText(x_train)
x_test=cleanText(x_test)

n_dim=120

imdb_w2v=Word2Vec(size=n_dim,min_count=10)
imdb_w2v.build_vocab(x_train)

imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
  
def buildWordVector(text,size):
    vec=np.zeros(size).reshape((1,size))
    count=0
    for word in text:
        try:
            vec+=imdb_w2v[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        vec/=count
    return vec

train_vecs=np.concatenate([buildWordVector(z,n_dim) for z in x_train])
train_vecs=ppmi(train_vecs,verbose=False,eps=1e-7)
train_vecs=scale(train_vecs)

imdb_w2v.build_vocab(x_test)
imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
test_vecs=np.concatenate([buildWordVector(z,n_dim) for z in x_test])
test_vecs=scale(test_vecs)


clf_rbf=svm.SVC(kernel="sigmoid",probability=True)
clf_rbf.fit(train_vecs,y_train)
print("train accuracy:%.2f"%clf_rbf.score(train_vecs,y_train))
print("test accuracy:%.2f"%clf_rbf.score(test_vecs,y_test))

from sklearn.metrics import roc_curve,auc
pred_probas=clf_rbf.predict_proba(test_vecs)[:,1]
fpr,tpr,_=roc_curve(y_test,pred_probas)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,label="area=%.2f"%roc_auc)
plt.plot([0,1],[0,1],"k--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.legend(loc="lower right")
plt.show()


