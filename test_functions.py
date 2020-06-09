from my_module import stats_functions as stats
import numpy as np

# Test the Average
def test_avg(a_list):
    if stats.avg(a_list) == np.mean(a_list):
        print("The Function Works")
    else:
        print("Something is Wrong with the Math")


# Test the Standard Deviation
def test_sd(a_list):
    if stats.sd(a_list) == np.std(a_list):
        print("The Function Works")
    else:
        print("Something is Wrong with the Math")