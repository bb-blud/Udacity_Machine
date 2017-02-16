#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
##Assuming shell or file is in ~/ProjProg/Machine_Learning_Nano/P3_Unsupervised_learning/ud120-projects
import pickle
import re

#Get pickled data
enron_data = pickle.load(open("./final_project/final_project_dataset.pkl", "r"))
name_file = open("./final_project/poi_names.txt", "r")

#Read email text file
names = name_file.readlines()
names = [s for s in names if s[0] == '(']
print "Number of Pois: ", len(names)

#Find facts about Colwell wesley. Using regular expression to find exact key strings for example:
#[key for key in enron_data['COLWELL WESLEY'].keys() if re.search("poi", key)
stock = enron_data['COLWELL WESLEY']['total_stock_value']
colwell2Poi = enron_data['COLWELL WESLEY']['from_this_person_to_poi']

#[key for key in enron_data.keys() if re.search("SK", key)]
#[key for key in enron_data['SKILLING JEFFREY K'] if re.search("stock", key)]

ex_stock = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
