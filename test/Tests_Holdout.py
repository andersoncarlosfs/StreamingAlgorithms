
# coding: utf-8

# In[1]:


# %load ../code/adaptive_random_forests.py


# # Classifier

# ## Authors

# In[1]:


__author__ = 'Anderson Carlos Ferreira da Silva'


# ## Imports

# In[2]:


import sys
import logging
import math
import numpy as np
from random import randint
from operator import attrgetter
from skmultiflow.core.utils.utils import *
from skmultiflow.core.base_object import BaseObject
from skmultiflow.classification.base import BaseClassifier
from skmultiflow.classification.trees.hoeffding_tree import *
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
#from joblib import Parallel, delayed
from adaptive_random_forests_not_parallel import AdaptiveRandomForest


# ## Constants

# In[3]:


POISSON_SIZE = 1
INSTANCE_WEIGHT = np.array([1.0])
FEATURE_MODE_M = ''
FEATURE_MODE_SQRT = 'sqrt'
FEATURE_MODE_SQRT_INV = 'sqrt_inv'
FEATURE_MODE_PERCENTAGE = 'percentage'


# ## ADFHoeffdingTree

# ### References

# - [Hoeffding Tree](https://github.com/scikit-multiflow/scikit-multiflow/blob/17327dc81b7d6e35d533795ae13493ad08118708/skmultiflow/classification/trees/hoeffding_tree.py)
# - [Adaptive Random Forest Hoeffding Tree](https://github.com/Waikato/moa/blob/f5cdc1051a7247bb61702131aec3e62b40aa82f8/moa/src/main/java/moa/classifiers/trees/ARFHoeffdingTree.java)

# In[4]:


def attribute_observers(i, list_attributes, _attribute_observers, X, y, weight, ht):    
    obs = _attribute_observers[i]
    if obs is None:
        if i in ht.nominal_attributes:
            obs = NominalAttributeClassObserver()
        else:
            obs = GaussianNumericAttributeClassObserver()
        _attribute_observers[i] = obs
    obs.observe_attribute_class(X[i], int(y), weight)

# # Tests for ARF.
# 
# ## For the following part we test the ARF algorithm according to several parameters:
#     -If the stream is created with a csv file or with the waveformgenerator()
#     
#     -We test for 3 datasets: covtype, movingSquares and sea_stream.
#     
#         -covtype: The forest covertype data set represents forest cover type for 30 x30 meter cells obtained from the US Forest Service Region 2 Resource InformationSystem (RIS) data. Each class corresponds to a diﬀerent cover type. This dataset contains 581,012 instances, 54 attributes (10 numeric and 44 binary) and 7imbalanced class labels. 
#        
#        -movingsquares:
#         
#        -sea_stream: The SEA generator produces data streams with three continuousattributes (f1, f2, f3). The range of values that each attribute can assume is be-tween 0 and 10. Only the ﬁrst two attributes (f1, f2) are relevant, i.e., f3doesnot inﬂuence the class value determination. New instances are obtained throughrandomly setting a point in a two dimensional space, such that these dimensionscorresponds to f1and f2. This two dimensional space is split into four blocks,each of which corresponds to one of four diﬀerent functions. In each block a pointbelongs to class 1 if f1+f2≤θand to class 0 otherwise. The threshold θusedto split instances between class 0 and 1 assumes values 8 (block 1), 9 (block 2), 7(block 3) and 9.5 (block 4). It is possible to add noise to class values, being thedefault value 10%, and to balance the number of instances of each class. SEAgsimulates 3 gradual drifts, while SEAasimulates 3 abrupt drifts 
# 
# Adaptive random forests for evolving data stream classification (PDF Download Available). Available from: https://www.researchgate.net/publication/317579226_Adaptive_random_forests_for_evolving_data_stream_classification [accessed Feb 11 2018].
#        
#     -We use 2 evaluators: Prequential and Holdout. Hold-out is more accurate, but needs data for testing.
#     
#         -Prequential:The error of a model is computed from the sequence of examples. 
#         For each example in the stream, the actual model makes a prediction, and then uses it to update the model.
#         
#         -Holdout:Apply the current decision model to the test set,at regular time intervals 
#         The loss estimated in the hold out is an unbiased estimator
#         
#     -Finally we compare the model with a naive bayes model, BernoulliNB, with the accuracy and the Kappa statistic.
#         
#         -Accuracy:
#         
#         -Kappa statistic: k=1 the classifier is always correct. 
#                           k=0 the predictions coincide with the correct ones as often as those of the chance classifier
#         
#         
# ADFHoeffdingtree with wave form generator stream

# In[ ]:


from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_holdout import EvaluateHoldout


# 1. Create a stream
stream = WaveformGenerator()
stream.prepare_for_use()

# 2. Instantiate the classifier
adf = AdaptiveRandomForest()

# 3. Setup the evaluator
eval = EvaluateHoldout(show_plot=True, pretrain_size=100, max_instances=10000)

# 4. Run evaluation
eval.eval(stream=stream, classifier=adf)


# 
# # Eval Holdout with datasets.csv for ARF 
# 

# In[ ]:


from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_holdout import EvaluateHoldout


# 1. Create a stream
#options = FileOption(option_value="../datasets/covtype.csv", file_extension="CSV")
options = FileOption(option_value="../datasets/movingSquares.csv", file_extension="CSV")
#options = FileOption(option_value="../datasets/sea_stream.csv", file_extension="CSV")

stream = FileStream(options)

stream.prepare_for_use()

# 2. Instantiate the classifier
adf = AdaptiveRandomForest()

# 3. Setup the evaluator
eval = EvaluateHoldout(pretrain_size=200, max_instances=10000, batch_size=1, max_time=1000, output_file='resultsHoldout.csv', task_type='classification', show_plot=True, plot_options=['kappa', 'performance'], test_size=5000, dynamic_test_set=True)

# 4. Run evaluation
eval.eval(stream=stream, classifier=adf)


# # compare ARF AND NAIVE BAYES classifiers with holdout

# In[ ]:


# The second example will demonstrate how to compare two classifiers with
# the EvaluateHoldout
from skmultiflow.evaluation.evaluate_holdout import EvaluateHoldout

from sklearn.naive_bayes import BernoulliNB

options = FileOption(option_value="../datasets/covtype.csv", file_extension="CSV")
#options = FileOption(option_value="../datasets/movingSquares.csv", file_extension="CSV")
#options = FileOption(option_value="../datasets/sea_stream.csv", file_extension="CSV")
stream = FileStream(options)

stream.prepare_for_use()

clf_one = BernoulliNB()
clf_two = AdaptiveRandomForest()
classifier = [clf_one, clf_two]

eval = EvaluateHoldout(pretrain_size=200, test_size=5000, dynamic_test_set=True, max_instances=100000, batch_size=1,  max_time=1000, output_file='comparison_Bernoulli_ADFH_Holdout.csv', task_type='classification', show_plot=True, plot_options=['kappa', 'performance'])
eval.eval(stream=stream, classifier=classifier)


# In[ ]:


# Verification of the same result for ADFHoeffdingTree
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.options.file_option import FileOption
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree


# Setup the File Stream
opt = FileOption("FILE", "OPT_NAME", "../datasets/covtype.csv", "CSV", False)
#opt = FileOption("FILE", "OPT_NAME", "../datasets/movingSquares.csv", "CSV", False)
#opt = FileOption("FILE", "OPT_NAME", "../datasets/sea_stream.csv", "CSV", False)
stream = FileStream(opt, -1, 1)
stream.prepare_for_use()

# Setup the classifiers
clf_one = HoeffdingTree()
clf_two = AdaptiveRandomForest()

# Setup the pipeline for clf_one
pipe = Pipeline([('Classifier', clf_one)])

# Create the list to hold both classifiers
classifier = [pipe, clf_two]

# Setup the evaluator
eval = EvaluateHoldout(pretrain_size=200, max_instances=100000, batch_size=1, max_time=1000, output_file='comparison_Hoeffding_ADFH_Preq.csv', task_type='classification', plot_options=['kappa', 'kappa_t', 'performance'])

# Evaluate
eval.eval(stream=stream, classifier=classifier)

