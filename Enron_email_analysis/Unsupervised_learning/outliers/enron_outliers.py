# coding: utf-8

# In[4]:

#!/usr/bin/python
import numpy as np
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../../final_project/final_project_dataset.pkl", "r") )

# Remove TOTAL key from data
data_dict.pop('TOTAL',0)


features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

print data.shape


# In[5]:

### Scatter plot data
get_ipython().magic(u'matplotlib inline')

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# In[6]:

## Who's the outlier
def findOutliers():
    dd = data_dict
    # Take out NaN's
    bad_keys = [k for k in dd.keys() if dd[k]['bonus'] == 'NaN' or dd[k]['salary'] == 'NaN']
    # Find outliers
    return [k for k in dd.keys() if k not in bad_keys and dd[k]['bonus'] > 5e6 and dd[k]['salary'] > 1e6]

print findOutliers()


# In[11]:

#help(np.dot)
print np.dot(data[0],data[0])
k = data_dict.keys()[0]
print data_dict[k]['bonus']**2 + data_dict[k]['salary']**2
print type( (data_dict[k]['salary']**2 + data_dict[k]['bonus']**2)**0.5 )
# def findOutlier():
#     # Find longest vector
#     max_d = max( [ np.dot(u,u) for u in data] )

#     # Find corresponding key for that vector
#     for key in data_dict.keys():
# 	salary = data_dict[key]['salary']
# 	bonus  = data_dict[key]['bonus']
# 	if salary != 'NaN' and bonus != 'NaN' and abs(salary**2 + bonus **2 - max_d) < 0.01:
# 	    return key

