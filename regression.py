import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    #response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    from scipy import stats
    train_data=[]
    train_label=[]
    with open('linreg_train.csv') as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        i=0
        for row in readCSV:
            if i==0:
                train_data.append(row[1:])
                i=i+1
            else:
                train_label.append(row[1:])      
    train_data=numpy.reshape(train_data,len(train_data[0]))
    train_label=numpy.reshape(train_label,len(train_label[0]))
    train_label=[float(i) for i in train_label]
    train_data=[float(i) for i in train_data]

    slope,intercept,r_value,p_value,std_err=scipy.stats.linregress(train_data,train_label)

    price=intercept+slope*area
    return price


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
