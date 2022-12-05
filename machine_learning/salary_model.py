import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import numpy as np
from scipy.stats import zscore
import os
import tensorflow as tf
import statistics
from sklearn import metrics
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping


"""
This code outlines some of the formatting performed on the data to prepare it for training.

It also includes the code for the model.

NOTE: this code was written in a different environment than the one the application was written 
      in. Chances are that running it as is will not work. It is included so that you can 
      see how we developed our salary prediction model. Namely, the steps used in formatting the data
      and the general model strucutre.
"""

# keep in mind to remove the original categorical data in the orginal dataframe after this function
# returns the encoded data, then concatenate this data to it.
"""
Function: dummy_one_hot_encoding()

Purpose: This function performs hot-one-encoding for categorical values in data. Given a dataframe, target
         column name, and a prefix. It will create a list of unique categorical values within a category 
         specified by column_name. It will then call the pd.get_dummies function to create dummy variables.
         It returns these dummy varaibles so that they can be concatenated to the original data again.

Parameters: 
    DataFrame df
    String column_name
    String pref

Return:
    DataFrame dummies
"""
def dummy_one_hot_encoding(df, column_name, pref="dummy"):
    
    # get unique categorical values. 
    #values = list(df[column_name].unique())
    #print(f"Number of areas: {len(values)}")
    #print(f"Values {values}\n")

    # we may want to add something for sorting the data here later, just for viewing
    # purposes.

    # make dummy vars, a column for each value in the values list
    dummies = pd.get_dummies(df[column_name], prefix=pref)
    #print(dummies[0:10])

    #drop the first column, technically more efficient.
    #dummies = pd.get_dummies(values,prefix=pref, drop_first=True)

    # return dummies variables to be concatenated back to the data
    return dummies

"""
Function: calc_smooth_mean()

Purpose:

Parameters:

Return:
"""
def calc_smooth_mean(df1, df2, global_mean, counts_agg, means_agg, category, target, weight):
    # Compute the "smoothed" means
    smooth = (counts_agg * means_agg + weight * global_mean) / (counts_agg + weight)
    #print(smooth)
    return smooth
     # Replace each value by the according smoothed mean
    #if df2 is None:
        #return df1[category].map(smooth)
    #else:
        #return df1[category].map(smooth),df2[category].map(smooth.to_dict())

def plot_loss(history, axs, x_label="epoch", y_label="y"):
    
    format_chart(axs)

    axs.plot(history.history['loss'], label='loss', color="red")
    axs.plot(history.history['val_loss'], label='val_loss', color="#4B90F8")
    
    axs.set_xlabel(x_label, labelpad=15, color='#000000', weight='bold')
    axs.set_ylabel(y_label, labelpad=15, color='#000000', weight='bold')
    axs.set_title("Loss Plot", pad=15, color='#000000',weight='bold')
    axs.legend()
    axs.grid(True)

def chart_regression(prediction, y, axs, sort=True):

    #display(prediction)

    print("check 1")
    
    prediction_y_df = pd.DataFrame()  
    prediction_y_df["prediction"] = prediction
    prediction_y_df["y"] = y.flatten()
    display(prediction_y_df)
    

    if sort:
        prediction_y_df.sort_values(by=["y"], inplace=True)

    #fig, axs = plt.subplots(1, figsize=(12,14))
    
    format_chart(axs)

    axs.plot(prediction_y_df["y"].tolist(), label="expected", color="red")
   
    axs.plot(prediction_y_df["prediction"].tolist(), label="prediction", color="#4B90F8")
    # set labels and format them.
    axs.set_xlabel('Elements Tested', labelpad=15, color='#000000', weight='bold')
    axs.set_ylabel('Output Median Income ($CAD)', labelpad=15, color='#000000', weight='bold')
    axs.set_title('Expected Values vs Predicted Values for Median Income', pad=15, color='#000000',
             weight='bold')
    axs.legend()

    #fig.tight_layout()
    return None

"""
Function: DNN()

Purpose: This function is used to build the predictive model. It takes a Dataframe which has been formatted and had its values
         encoded and normalized. The Median income field associated with this Dataframe becomes our target, the remaining values 
         are used for predicting this target and make up our vector space. We create train test splits using a random seed.
         We use a sequential model with dense layers (fully connected). The activation function is relu for both hidden layers.
         The optimizer is adam. We implement an early stoppping method to prevent over fitting.

Parameters:
    df: a Dataframe containing the target value and the vectors associated with them.

Return:
    None

Sources:
    # https://www.tensorflow.org/tutorials/keras/regression
    # https://keras.io/api/callbacks/early_stopping/
"""
def DNN(df) -> None:

    # predictors from df to numpy array	
    # #"cred_Bachelor's degree", "cred_Certificate", "cred_Diploma", "cred_Doctoral degree", "cred_Master's degree", "cred_Professional bachelor's degree", "Field of Study (2-digit CIP code)"		
    x = df[["Years After Graduation", "Field of Study (2-digit CIP code)", "cred_Bachelor's degree", "cred_Certificate", "cred_Diploma", "cred_Doctoral degree", "cred_Master's degree", "cred_Professional bachelor's degree"]].values
    # Target value
    y = df["Median Income"].values

    #split data for training
    x_train, x_test, y_train, y_test = train_test_split(    
                                    x, 
                                    y, 
                                    test_size=0.25, 
                                    random_state=42)

    # Build the neural network
    salary_model = Sequential()
    salary_model.add(Dense(25, input_dim=x.shape[1], activation='relu')) # Hidden 1
    salary_model.add(Dense(10, activation='relu')) # Hidden 2
    salary_model.add(Dense(1)) # Output
    salary_model.compile(loss='mean_squared_error', optimizer="adam")

    # defines early stopping requirements and parameters.
    callback = EarlyStopping(
        # monitor is the quantity being monitored.
        monitor='val_loss', 
        # min_delta is the minimum change in the mointored quntity to qualify as an improvement.
        min_delta=1e-2,
        # number of epochs to wait before stopping upon no improvement.
        patience=10, 
        # Logging level, 0 for none.
        verbose=1, 
        mode='auto',
        # restore the weights as they were number of patience epochs ago.
        restore_best_weights=True)

    # train the neural network
    history = salary_model.fit(
                # Training x values   
                x_train, 
                # Training y values
                y_train,
                validation_data=(x_test,y_test),
                callbacks=[callback],
                verbose=2,
                epochs=1000)

    # Run the now trained model on the test data. Notice we ware using x_test, where
    # above in the fitting of the model we used x_train. Predict will return a series of predictions.
    test_predictions = salary_model.predict(x_test)
    
    # Measure Mean Squared Error (MSE); the MSE is the sum of the squared differences between 
    # some predcited value in its counterpart expected value. It basically measures, how much
    # the predictions deviate from the actual values on average. It is always positive and a lower
    # MSE is ideal. There isn't much you can gather from MSE, it has no meaningful units.
    # score = metrics.mean_squared_error(test_predictions,y_test)

    # Measure RMSE error. RMSE is common for regression. Is the root of the MSE, this 
    # puts the RSME error in the same units as the target value.
    # pass it a series of the predictions, and their corresponding actual scores (y_test)
    score = np.sqrt(metrics.mean_squared_error(test_predictions,y_test))
    print("(RMSE): " + str(score))

    # save the model so we do not have to continuously re-train it.
    # salary_model.save(os.path.join(".","Salary_Model.h5"))

    fig, axs = plt.subplots(2, figsize=(12,14))

    # Graph the loss and regression outcomes.
    plot_loss(history, axs[0], x_label="epoch", y_label="y")
    chart_regression(pd.Series(test_predictions.flatten()), y_test, axs[1], sort=True) #flattened

    fig.tight_layout()
   
    return 0

"""
Function: prep_vector_space()

Purpose: before passing data to a neural network, it must be altered such that a 
         neural network can use it. Here we encode and normalize the values passed in 
         the dataframe.

Parameters:
    df: the Dataframe we are going to modify such that a neural network
        can use the values with-in it.
Returns:
    df3: the modified dataframe.
"""
def prep_vector_space(df):

    # Compute the global mean based on target 
    mean = df["Median Income"].mean()
    # Compute the number of values and the mean of each group
    agg = df.groupby("Field of Study (2-digit CIP code)")["Median Income"].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    # calculate smoothed means
    smoothed_means = calc_smooth_mean(df1=df, df2=None, global_mean=mean, counts_agg=counts, means_agg=means, category="Field of Study (2-digit CIP code)", target="Median Income", weight=2)
    # convert to dictionary
    smoothed_means_dict = smoothed_means.to_dict()

    print(smoothed_means_dict)
    
    tempa = list(smoothed_means_dict.values())
    tempb = list(smoothed_means_dict.keys())
    print(tempb)

    std = statistics.stdev(tempa)
    mn = statistics.mean(tempa)

    print(f"std:{std} and mean:{mn}")

    
    # replace with smoothed means
    df1  = df.replace({"Field of Study (2-digit CIP code)": smoothed_means_dict})
    
    # UNCOMMENT TO SEE VALUE RANGE BEFORE NORMALIZATION
    #i=df3["Field of Study (2-digit CIP code)"].min()
    #l=df3["Field of Study (2-digit CIP code)"].max()
    #print(i)
    #print(l)
    # Normalize "Field of Study (2-digit CIP code)"
    df1["Field of Study (2-digit CIP code)"] = zscore(df1["Field of Study (2-digit CIP code)"])
    # UNCOMMENT TO SEE VALUE RANGE AFTER NORMALIZATION
    #i=df3["Field of Study (2-digit CIP code)"].min()
    #l=df3["Field of Study (2-digit CIP code)"].max()
    #print(i)
    #print(l)

    # Normalize "Years After Graduation"
    df1["Years After Graduation"] = zscore(df1["Years After Graduation"])

    # remove the Field of Study (4-digit CIP code) column from the dataset
    df1 = df1.drop(columns="Field of Study (4-digit CIP code)", axis=1)
    
    dummies = dummy_one_hot_encoding(df1, "Credential", pref="cred")
    #print(dummies)
    
    #concatenate our dummy variables
    df2 = pd.concat([df1,dummies],axis=1)
    #drop the old credentials column
    df2 = df2.drop(columns="Credential", axis=1)
    #drop the cohort column
    df2 = df2.drop(columns="Cohort Size", axis=1)

    #shuffle values
    df3 = df2.reindex(np.random.permutation(df2.index))
    #fix indexes
    df3 = df3.reset_index(inplace=False, drop=True)

    # format the float values of this column to two decimal places.
    df3["Field of Study (2-digit CIP code)"] = df3["Field of Study (2-digit CIP code)"].map('{:.2f}'.format)
    df3["Years After Graduation"] = df3["Years After Graduation"].map('{:.3f}'.format)
    #save_df_to_csv(df3, "alberta_salary_data_edited.csv", path=".")

    #display(df3)

    return df3

"""
Function: format_chart()

Purpose: applies a format to the charts that we output such that they are consistent.

Parameters:
    ax: axis

Return:
    None

Sources:
    # https://www.pythoncharts.com/matplotlib/beautiful-bar-charts-matplotlib/
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
"""
def format_chart(ax) -> None:

    # remove unecessary lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    # remove ticks
    ax.tick_params(bottom=False, left=False)
    # Add a horizontal grid. Color the lines a light gray.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    # get the color of the bars.

    return None

"""
Function: format_data()

Purpose: do some preliminary formatting of the data.

Parameters:
    df: the Dataframe we are manipulating.

Return:
    df1: the manipulated Dataframe.
"""
def format_data(df):

    # before drop 2480 rows
    df1 = df.dropna(axis=0)
    #after 1670 rows
    #print(len(df))

    # remove un-wanted characters from median income field
    df1['Median Income'] = df1['Median Income'].str.replace('$','')
    df1['Median Income'] = df1['Median Income'].str.replace(',','')
    # convert median income string to a numeric value
    df1["Median Income"] = pd.to_numeric(df1["Median Income"])
    # The data is grouped, we don't want this
    #df = df.reindex(np.random.permutation(df.index))

    return df1

"""
Function: save_df_to_csv()

Purpose: save the a dataframe to a csv file format

Parameters:
    df: Dataframe being saved
    file_name: the filename
    path: the path.

Returns:
    None
"""
def save_df_to_csv(df, file_name, path=".") -> None:
    write_file = os.path.join(path, file_name)
    df = df.reindex(np.random.permutation(df.index))
    # Specify index = false to not write row numbers
    df.to_csv(write_file, index=False)
    print("completed")
    return None

"""
This is the start of the program.
"""
def main() -> None:

    df = read_csv_to_df("graduate_earnings.csv",verbose=0,rows=30)

    df = format_data(df)

    # Prepare the vector space based on analysis of features.
    reformed_data = prep_vector_space(df)

    DNN(reformed_data)

    return None

# program driver
if __name__ == "__main__":
    main()

