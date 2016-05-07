import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 200)

df = pd.read_csv(os.path.join(os.path.dirname(__name__), 'iris.csv'))

species = np.array(df.species)
params = np.array(df.drop('species', axis=1))

# we should probably ensure that all species appear in train _and_ test
params_train, params_test, species_train, species_test = train_test_split(
    params,
    species,
    test_size=0.4,
    random_state=0)

scaler = StandardScaler()
scaler.fit(params_train)
params_train_scaled = scaler.transform(params_train)
params_test_scaled = scaler.transform(params_test)

mlp = MLPClassifier(
    algorithm='l-bfgs',
    hidden_layer_sizes=(df.shape[1] - 1, df.shape[1] - 1),
    random_state=0)
mlp.fit(params_train_scaled, species_train)

print 'Train score: %s / Test Score: %s' % (
    mlp.score(params_train_scaled, species_train),
    mlp.score(params_test_scaled, species_test))
print

predicted = pd.DataFrame(mlp.predict(params_test_scaled), columns=['predicted'])
params_df = pd.DataFrame(params_test, columns=df.drop('species', axis=1).columns)
actual = pd.DataFrame(species_test, columns=['actual'])
concat = pd.concat((predicted, actual), axis=1)

failed = params_df.join(concat[concat.predicted != concat.actual], how='inner')

if failed.empty:
    print 'Classified all test samples correctly!'

else:
    failed_array = np.array(
        failed.drop('actual', axis=1).drop('predicted', axis=1))
    failed_array_scaled = scaler.transform(failed_array)
    failed_with_proba = pd.DataFrame(
        mlp.predict_proba(failed_array_scaled),
        columns=mlp.classes_)

    print 'Failed to classify the following test samples:'
    print pd.concat((failed.reset_index(drop=True), failed_with_proba), axis=1)
    print

# layer #0 being the input layer
for i, weight_matrix in enumerate(mlp.coefs_, start=1):
    print 'Layer #%d' % i

    if i == 1:
        weight_matrix_df = pd.DataFrame(weight_matrix.transpose())
        weight_matrix_df.columns= df.drop('species', axis=1).columns

    else:
        weight_matrix_df = pd.DataFrame(weight_matrix.transpose())
        weight_matrix_df.columns = ['Neuron %d from layer %d' % (c, i - 1)
            for c in weight_matrix_df.columns]

    weight_matrix_df.index = weight_matrix_df.index.map(lambda x: 'Neuron #%d' % x)
    print weight_matrix_df

    print
