from time_series_splitter import TimeSeriesSplitter
import numpy as np

array2d = np.arange(4000).reshape((1000, 4))
array1d = np.arange(1000)

tss = TimeSeriesSplitter(x=array2d, y=array1d)

tss.split(test_periods=30, train_periods=365, n_validations=10, window_type='rolling')
tss.plot()

tss.split(test_periods=30, min_train_periods=365, n_validations=10, window_type='expanding')
tss.plot()
