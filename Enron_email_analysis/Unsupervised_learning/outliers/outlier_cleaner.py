#!/usr/bin/python
import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### Outlier Removal
    fraction = .10
    no_removals = int(len(ages)*fraction)

    ## Squared distances (errors)
    s_diffs = [d**2 for d in net_worths - predictions]

    ## Find indices of top 10% errors
    error_indices = []
    for k in range(no_removals):
        top_error = 0
        max_i = 0
        for i, err in enumerate(s_diffs):
            if err > top_error and i not in error_indices:
                top_error = err
                max_i = i
        error_indices.append(max_i)


    cleaned_data = [(age, net_worths[i], s_diffs[i]) for i, age in enumerate(ages) if i not in error_indices]
      
    return numpy.array(cleaned_data)

