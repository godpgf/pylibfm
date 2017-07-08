# Factorization Machines in Python

This is a python implementation of Factorization Machines [1]. This uses stochastic gradient descent(sgd) with adaptive regularization as a learning method, which adapts the regularization automatically while training the model parameters. The future will support "sgda" and "mcmc". See [2] for details. From libfm.org: "Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain."

[1] Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May.
[2] Steffen Rendle: Learning recommender systems with adaptive regularization. WSDM 2012: 133-142

## Installation
```
git clone https://github.com/godpgf/pylibfm.git
cd pylibfm
make all
```

## Dependencies
* numpy
* sklearn

## Training Representation
The easiest way to use this class is to represent your training data as lists of standard Python dict objects, where the dict elements map each instance's categorical and real valued variables to its values. Then use a [sklearn DictVectorizer](http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer) to convert them to a design matrix with a one-of-K or “one-hot” coding.

Here's a toy example

```python
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
train = [
    {"user": "1", "item": "5", "age": 19},
    {"user": "2", "item": "43", "age": 33},
    {"user": "3", "item": "20", "age": 55},
    {"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print(X.toarray())
y = np.array([float(i%2) for i in xrange(X.shape[0])])
fm = pylibfm.FM()
fm.fit(X,y)
print fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
```

## Getting Started
Here's an example on some real  movie ratings data.

First get the smallest movielens ratings dataset from http://www.grouplens.org/system/files/ml-100k.zip.
ml-100k contains the files u.item (list of movie ids and titles) and u.data (list of user_id, movie_id, rating, timestamp).

```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from sklearn.metrics import mean_squared_error

# Read in data
def loadData(filename,path="ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

(train_data, y_train, train_users, train_items) = loadData("ua.base")
(test_data, y_test, test_users, test_items) = loadData("ua.test")
v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

# Build and train a Factorization Machine
fm = pylibfm.FM(num_factor=8, num_iter=100, task="regression", learning_rate=0.001)

fm.fit(X_train,y_train)

preds = fm.predict(X_train)
print("Train FM MSE: %.4f" % mean_squared_error(y_train,preds))

preds = fm.predict(X_test)
print("Test FM MSE: %.4f" % mean_squared_error(y_test,preds))
```

## Classification example
```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from pyfm import pylibfm

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1)
data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factor=50, num_iter=100, task="classification", learning_rate=0.0001)

fm.fit(X_train,y_train)

# Evaluate
from sklearn.metrics import log_loss
print "Validation log loss: %.4f" % log_loss(y_test,fm.predict(X_test))
```
