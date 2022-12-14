import numpy
import matplotlib.pyplot


class TimeSeriesSplitter:
    """
    splits time series data into sequential train & validation pairs
    according to required inputs for train & validation length, window type and number of validations
    """
    training_validation_indices = None
    window_type = None
    n_validations = None
    min_train_periods = None
    training_indices = None
    validation_indices = None

    def __init__(self, x, y):
        """
        :param x: 1d or 2d array like object
        :param y: 1d array like object
        """
        if isinstance(x, numpy.ndarray):
            self.x_data = x
        elif isinstance(x, list):
            self.x_data = numpy.array(x)
        else:
            self.x_data = numpy.array(x.values)

        if isinstance(y, numpy.ndarray):
            self.y_data = y
        elif isinstance(x, list):
            self.y_data = numpy.array(y)
        else:
            self.y_data = numpy.array(y.values)

        if len(x) != len(y):
            raise ValueError('X and y must be of same length')
        else:
            self.rows = len(y)
            self.row_index = numpy.arange(self.rows)

    def validate_inputs(self, test_periods: int, train_periods: int, min_train_periods: int, n_validations: int,
                        window_type: str):
        """
        raises error if these inputs are incompatible compared to the overall rows of the data
        :param test_periods: the length of the testing period in units supplied (days, hours etc.)
        :param train_periods: the length of the training period in units supplied (days, hours etc.)
        (compatible with window_type='rolling')
        :param min_train_periods: starting length of first training period in units supplied (days, hours etc.)
        (compatible with window_type='expanding')
        :param n_validations: validations periods required. if None, this will be calculated based on other parameters
        :param window_type: either 'expanding' or 'rolling'
        'rolling' the training periods will be the same length for each validation set
        'expanding' the training periods will increase with each validation set to make the most of the data supplied
        """

        if (train_periods is not None) & (min_train_periods is not None):
            raise ValueError(
                'values cannot be assigned for both train_periods & min_train_periods'
            )

        if (train_periods is None) & (min_train_periods is None):
            raise ValueError('one of train_periods & min_train_periods must be assigned')

        if window_type not in ['rolling', 'expanding']:
            raise ValueError('training windows must either be "rolling" or "expanding"')

        if (train_periods is not None) & (window_type == 'expanding'):
            raise ValueError(
                'if window is expanding, min_train_periods should be assigned'
                ' train_periods is for a rolling window.'
            )

        if (min_train_periods is not None) & (window_type == 'rolling'):
            raise ValueError(
                'if window is rolling, train_periods should be assigned'
                ' min_train_periods is for an expanding window.'
            )
        if min_train_periods is not None:
            if min_train_periods <= 1:
                raise ValueError('min_train_periods must be at least 2')
        if train_periods is not None:
            if train_periods <= 1:
                raise ValueError('train_periods must be at least 2')
        if n_validations is not None:
            if n_validations == 0:
                raise ValueError('n_validations must be at least 1')

        if n_validations is not None:
            min_train_periods = min_train_periods if min_train_periods is not None else train_periods
            if n_validations * test_periods + min_train_periods > self.rows:
                raise ValueError(
                    'the data set is not large enough based on the requirements for'
                    ' the number of validations and size of test_periods & train periods.'
                )

    def split(self, test_periods: int, train_periods: int = None, min_train_periods: int = None,
              n_validations: int = None, window_type: str = 'expanding'):
        """
        calculates indices for splitting timeseries data
        :param test_periods: the length of the testing period in units supplied (days, hours etc.)
        :param train_periods: the length of the training period in units supplied (days, hours etc.)
        (compatible with window_type='rolling')
        :param min_train_periods: starting length of first training period in units supplied (days, hours etc.)
        (compatible with window_type='expanding')
        :param n_validations: validations periods required. if None, this will be calculated based on other parameters
        :param window_type: either 'expanding' or 'rolling'
        'rolling' the training periods will be the same length for each validation set
        'expanding' the training periods will increase with each validation set to make the most of the data supplied
        """

        self.validate_inputs(test_periods, train_periods, min_train_periods, n_validations, window_type)

        # calculate min train periods if not provided
        if min_train_periods is None:
            min_train_periods = train_periods

        # calculate n_validations if not provided
        if n_validations is None:
            n_validations = numpy.int64(numpy.floor((self.rows - min_train_periods) / test_periods))

        # configure training windows
        val_starts = list(self.row_index)[-test_periods:: -test_periods][: n_validations]
        val_ends = list(self.row_index)[:: -test_periods][: n_validations]
        val_ends = [ve + 1 for ve in val_ends]
        validation_indices = list(zip(val_starts, val_ends))
        validation_indices.reverse()

        # adjusting min train period up so that we don't have any gaps not being used for training or validation
        # OR fixing to train_periods input
        first_val_loc = list(self.row_index).index(min(val_starts))
        min_train_periods = self.row_index[0: first_val_loc].size if train_periods is None else train_periods

        # configure training windows
        train_ends = list(self.row_index)[-test_periods:: -test_periods][: n_validations]

        if window_type == 'expanding':
            train_starts = [self.row_index.min() for i in range(len(train_ends))]
        elif window_type == 'rolling':
            train_starts = list(self.row_index)[-test_periods - min_train_periods:: -test_periods][: n_validations]

        training_indices = list(zip(train_starts, train_ends))
        training_indices.reverse()

        training_validation_indices = list(zip(training_indices, validation_indices))

        self.validation_indices = validation_indices
        self.training_indices = training_indices
        self.n_validations = n_validations
        self.min_train_periods = min_train_periods
        self.window_type = window_type
        self.training_validation_indices = training_validation_indices

    @property
    def training_validation_data(self) -> list:
        """
        :return: list of tuples containing array of values for each training and validation pair
        """
        training_validation_data = list(range(self.n_validations))
        for count in range(self.n_validations):
            val_start, val_end = self.validation_indices[count]
            x_val_data = self.x_data[val_start: val_end]
            y_val_data = self.y_data[val_start: val_end]

            train_start, train_end = self.training_indices[count]
            x_train_data = self.x_data[train_start: train_end]
            y_train_data = self.y_data[train_start: train_end]

            training_validation_data[count] = tuple((x_train_data, x_val_data, y_train_data, y_val_data))

        return training_validation_data

    def plot(self):
        """
        :return: plots a Gantt style graph to help visualise each training & validation set
        """

        # Declaring a figure "gnt"
        fig, gnt = matplotlib.pyplot.subplots()

        for i in range(self.n_validations):
            val_indices = self.validation_indices[i]
            gnt.broken_barh([(val_indices[0], val_indices[1] - val_indices[0])], (i, 1),
                            facecolors='orange'
                            )

            train_indices = self.training_indices[i]
            gnt.broken_barh([(train_indices[0], train_indices[1] - train_indices[0])], (i, 1),
                            facecolors='red'
                            )
        gnt.set_xlabel('time')
        gnt.set_ylabel('validations')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        title = f'timeseries cv with {self.window_type} window' \
                f' and {self.n_validations} validations'
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.show()
