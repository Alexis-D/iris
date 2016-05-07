import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 200)
pd.set_option(
    'display.float_format',
    lambda x: ('%.7f' % x).rstrip('0').rstrip('.'))

df = pd.read_csv(os.path.join(os.path.dirname(__name__), 'iris.csv'))

species = df[['species']]
params = df.drop('species', axis=1)

# we should probably ensure that both the training set and the test set both
# contain all 3 species. but given that all species have 50 samples it's
# unlikely that it's going to be a problem for now
params_train, params_test, species_train, species_test = train_test_split(
    params,
    species,
    # a 80/20 split is more common, but we want to make sure that the 'failed'
    # prediction code path works (which also explains the hard coded
    # random_state)
    test_size=0.4,
    random_state=0)

# scale the parameters, as advised here:
# http://scikit-learn.org/dev/modules/neural_networks_supervised.html#tips-on-practical-use
scaler = StandardScaler()
scaler.fit(params_train)
params_train_scaled = pd.DataFrame(
    scaler.transform(params_train),
    columns=params_train.columns,
    index=params_train.index)
params_test_scaled = pd.DataFrame(
    scaler.transform(params_test),
    columns=params_test.columns,
    index=params_test.index)

# 1 hidden layer, same size as the input layer
mlp = MLPClassifier(
    algorithm='l-bfgs',
    hidden_layer_sizes=(params.shape[1],),
    random_state=0)
mlp.fit(params_train_scaled, species_train.species)

print 'Train score: %.3g' % mlp.score(params_train_scaled, species_train)
print 'Test Score: %.3g' % mlp.score(params_test_scaled, species_test)
print

predicted = pd.DataFrame(
    mlp.predict(params_test_scaled),
    columns=['predicted'],
    index=params_test_scaled.index)
actual = species_test.rename(columns={'species': 'actual'})
concat = pd.concat((actual, predicted), axis=1)

failed = params_test.join(
    concat[concat.predicted != concat.actual],
    how='inner')

if failed.empty:
    print 'Classified all test samples correctly!'
    print

else:
    failed_params_scaled = scaler.transform(
        failed.drop('actual', axis=1).drop('predicted', axis=1))
    failed_with_proba = pd.DataFrame(
        mlp.predict_proba(failed_params_scaled),
        columns=mlp.classes_,
        index=failed.index)

    print 'Failed to classify the following test samples:'
    print pd.concat((failed, failed_with_proba), axis=1)
    print

# display the network weights for each layer rows in each table represent a
# neuron, and columns represent the input from the previous layer, a cell
# gives us the weight to use for a given neuron + input value
print 'Network weights, w/o bias, input layer is layer #0:'
for i, weight_matrix in enumerate(mlp.coefs_, start=1):
    print 'Layer #%d' % i

    if i == 1:
        weight_matrix_df = pd.DataFrame(weight_matrix.transpose())
        weight_matrix_df.columns = df.drop('species', axis=1).columns

    else:
        weight_matrix_df = pd.DataFrame(weight_matrix.transpose())
        weight_matrix_df.columns = ['Neuron #%d from layer #%d' % (c, i - 1)
                                    for c in weight_matrix_df.columns]

    weight_matrix_df.index = weight_matrix_df.index.map(
        lambda x: 'Neuron #%d' % x)
    print weight_matrix_df

    print
