'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Hannah Soria
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        data = self.data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        select_data = self.data.select_data(headers, rows)
        min = np.amin(select_data, 0)
        mins = np.array(min)
        return mins


    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        select_data = self.data.select_data(headers, rows)
        max = np.amax(select_data, 0)
        maxs = np.array(max)
        return maxs

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        return (self.min(headers, rows), self.max(headers, rows))

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''
        select_data = self.data.select_data(headers, rows)
        total_rows = select_data.shape[0]
        total = np.sum(select_data,0)
        mean = total/total_rows
        means = np.array(mean)
        return means
        

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        select_data = self.data.select_data(headers, rows) #select the data
        total_rows = select_data.shape[0] #number of item in the data
        mean = self.mean(headers, rows) # find the mean
        means = np.array(mean) # array of the means
        deviations = np.array((means - select_data)) # find each scores deviation from the mean
        squared = np.array(np.square(deviations)) # square each deviation from the mean
        total = np.sum(squared,0) # find the sum of the squares
        vars = total / (total_rows - 1) # divide the sum of the squares by N
        return vars

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        select_data = self.data.select_data(headers, rows) #select the data
        return np.sqrt(self.var(headers,rows))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        headers = [ind_var,dep_var] # independent and dependent variable as headers (x,y)
        rows = [] # for the rows data
        select_data = self.data.select_data(headers, rows) #select the data
        x = select_data[:,0] # the Xs are the first column
        y = select_data[:,1] # the Ys are the second column
        plt.figure(figsize=(10,10))
        plt.scatter(x,y) # create the plot
        plt.title(title) # set the title as the title passed
        plt.xlabel(ind_var) # set the x label as independent
        plt.ylabel(dep_var) # set the y label as the dependent
        return x,y


    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        fig, ax = plt.subplots(len(data_vars), len(data_vars), sharex='col', sharey='row') # create the subplots len(data_vars) x len(data_vars)
        fig.suptitle(title)
        fig.set_figwidth(fig_sz[0])
        fig.set_figheight(fig_sz[1])
        select_data = self.data.select_data(data_vars) # select the data
        for x in range(len(data_vars)): #iterate through the rows
            x_val = select_data[:,x] # set x val
            for y in range(len(data_vars)): # iterate through the cols
                y_val = select_data[:,y] # set y val
                ax[x,y].scatter(y_val,x_val) #create plot
                if x==len(data_vars)-1: # if at the last row add a label
                    ax[x,y].set_xlabel(data_vars[y]) 
                if y==0:    # if at first col add label
                    ax[x,y].set_ylabel(data_vars[x])
                # ax[x,y].set_xticks([]) #no ticks
                # ax[x,y].set_yticks([]) #no ticks
        plt.tight_layout(pad=2.0)
        return fig, ax
        
    # Extension 1
    def boxplot(self,var,ylabel='',title=''):
        headers = [var] # set header to var
        rows = [] # for the rows data
        select_data = self.data.select_data(headers, rows) #select the data
        x = select_data[:,0] # the Xs are the first column
        plt.xlabel(var) # label x axis
        plt.ylabel(ylabel) # label y axis
        plt.title(title) #make title
        plt.boxplot(x) # plot

    # Extension 2
    def iqr(self, header, rows=[]):
        headers = [header] #set headers
        select_data =np.array(self.data.select_data(headers, rows)) #select the data
        q1 = np.percentile(select_data, 25) # 25th percentile
        q3 = np.percentile(select_data, 75) # 75th percentile
        iqr = q3 - q1 # find IQR
        return iqr

    def hist(self,var,ylabel='',title=''):
        headers = [var] #set header
        rows = [] # for the rows data
        select_data = self.data.select_data(headers, rows) #select the data
        x = select_data[:,0] # the Xs are the first column
        plt.xlabel(var) # x label
        plt.ylabel(ylabel) # y label
        plt.title(title) # title
        plt.hist(x)
