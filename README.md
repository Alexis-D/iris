This is a quick weekend experiment with
[scikit-learn](http://scikit-learn.org/) and
[pandas](http://pandas.pydata.org/).

The script in `iris.py` uses a Neural Network to attempt classifying different
species of Iris given some of their features, for more information on this
dataset, see [Wikipedia: _Iris_ flower data
set](https://en.wikipedia.org/wiki/Iris_flower_data_set).

The version of `scikit-learn` being used hasn't been released yet so it was
installed using `pip install
git+git://github.com/scikit-learn/scikit-learn@master` on May 7th 2016. Once
`0.18.0` is released, it will be possible to install it directly from Pypi.

A sample run of the scripts looks like this:

```
 (iris) ✓ 21:21 adaboville @ adoboville-mbp in ~/Documents/stuff/iris (master ±) $ python iris.py
Train score: 1
Test Score: 0.967

Failed to classify the following test samples:
     sepal_length  sepal_width  petal_length  petal_width           actual        predicted  Iris-setosa  Iris-versicolor  Iris-virginica
83              6          2.7           5.1          1.6  Iris-versicolor   Iris-virginica            0        0.0006655       0.9993345
133           6.3          2.8           5.1          1.5   Iris-virginica  Iris-versicolor            0                1               0

Network weights, w/o bias, input layer is layer #0:
Layer #1
           sepal_length  sepal_width  petal_length  petal_width
Neuron #0     1.0496505   -1.1067977     2.6011602    1.5085547
Neuron #1     0.2037369    -3.616114     8.0559253   10.8794488
Neuron #2     3.2601155   -5.1798374     5.8090034    5.2365081
Neuron #3    -4.3364418    1.6549037    -5.2657274   -5.7168433

Layer #2
           Neuron #0 from layer #1  Neuron #1 from layer #1  Neuron #2 from layer #1  Neuron #3 from layer #1
Neuron #0               -0.4291321               -7.8116798               -8.9299367               14.8605455
Neuron #1               -1.8515707               -2.5020988                7.7603237              -10.7842877
Neuron #2                 2.958553               10.5252698                2.6352628               -3.6991152

```
