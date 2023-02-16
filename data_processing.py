import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import os

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")

folder_dir = os.getcwd()
dataset_dir = "/dataset/mxmh_survey_results.csv"
file_path = "".join([folder_dir, dataset_dir])
file = pd.read_csv(file_path)

def return_file():
    return file

def data_cleaning(dff):
    df = dff
    df.columns = [col.upper().replace(" ", "_") for col in df.columns]  # Clean column names

    df = df.dropna()  # Drop all rows with null values

    df.drop(columns=['BPM', 'PERMISSIONS'], axis=1,
            inplace=True)

    df['AGE'] = df['AGE'].astype(int)
    df = df[df['AGE'] <= 70]

    df = df.applymap(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))
    df = df.applymap(lambda x: 0 if x == 'Never' else (
        1 if x == 'Rarely' else (2 if x == 'Sometimes' else (3 if x == 'Very frequently' else x))))

    music_effect = []
    for col in df["MUSIC_EFFECTS"]:
        if col == "Improve":
            music_effect.append(1)
        else:
            music_effect.append(0)
    df["MUSIC_EFFECTS"] = music_effect

    df['MH_SCORE'] = df['ANXIETY'] + df['DEPRESSION'] + df['INSOMNIA'] + df['OCD']
    clean_freq_cols = []
    for col in df.columns:
        if ("[" in str(col)) and ("]" in str(col)):
            col = col.replace("[", "").replace("]", "")
            clean_freq_cols.append(col)
        else:
            clean_freq_cols.append(col)
    df.columns = clean_freq_cols
    return df


df = data_cleaning(dff=file)


def dataframe_summary(dataframe, cat_threshold=10, card_threshold=20):
    # Categorical Variables
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_threshold and dataframe[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > card_threshold and str(dataframe[col].dtypes) in ["datetime", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical Variables
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"------------ DATAFRAME SUMMARY ------------\n",
          f"Observations: {dataframe.shape[0]}\n",
          f"Variables: {dataframe.shape[1]}\n",
          f'cat_cols: {len(cat_cols)}\n{cat_cols}\n',
          f'num_cols: {len(num_cols)}\n{num_cols}\n',
          f'cat_but_car: {len(cat_but_car)}\n{cat_but_car}\n',
          f'num_but_cat: {len(num_but_cat)}\n{num_but_cat}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = dataframe_summary(dataframe=df)


# Define cat_summary function to get summary of categorical variables
def cat_summary(dataframe, plot=False):
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.grid()
            plt.show()


cat_summary(df, plot=False)


# Define num_summary function to get summary of numerical variables
def num_summary(dataframe, plot=False):
    for numerical_col in num_cols:
        quantiles = [0, .25, .50, .75, 1.]
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            dataframe[numerical_col].hist()
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()


num_summary(df, plot=False)


# Target analysis according to categorical variables
def target_summary_with_cat(dataframe, target, plot=False):
    for categorical_col in cat_cols:
        print(categorical_col)
        print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                            "Count": dataframe[categorical_col].value_counts(),
                            "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n")

        if plot:
            sns.countplot(x=dataframe[categorical_col], data=dataframe, hue=target)
            plt.grid()
            plt.show()


target_summary_with_cat(df, "MUSIC_EFFECTS", plot=False)


# Target analysis according to numerical variables
def target_summary_with_num(dataframe, target, plot=False):
    for numerical_col in num_cols:
        if numerical_col != "MH_SCORE":
            print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

            if plot:
                sns.barplot(x=dataframe[numerical_col], y=dataframe[target])
                plt.grid()
                plt.show()


target_summary_with_num(df, "MUSIC_EFFECTS")


# Define outlier_thresholds function to get upper and lower bounds of numerical variables (Outlier analysis)
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Outlier analysis and suppression process
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


def feature_extraction(df):
    # Age bins
    df.loc[(df["AGE"] < 18), "NEW_AGE_CLASS"] = 1
    df.loc[(df["AGE"] >= 18) & (df["AGE"] < 21), "NEW_AGE_CLASS"] = 2
    df.loc[(df["AGE"] >= 21) & (df["AGE"] < 24), "NEW_AGE_CLASS"] = 3
    df.loc[(df["AGE"] >= 24) & (df["AGE"] < 30), "NEW_AGE_CLASS"] = 4
    df.loc[(df["AGE"] >= 30) & (df["AGE"] <= max(df["AGE"])), "NEW_AGE_CLASS"] = 5
    df["NEW_AGE_CLASS"] = df["NEW_AGE_CLASS"].astype("int64")

    df.loc[(df["HOURS_PER_DAY"] <= 0) & (df["HOURS_PER_DAY"] <= 2), "NEW_HOURS_PER_DAY_STAGE"] = 1
    df.loc[(df["HOURS_PER_DAY"] > 2) & (df["HOURS_PER_DAY"] <= 4), "NEW_HOURS_PER_DAY_STAGE"] = 2
    df.loc[(df["HOURS_PER_DAY"] > 4) & (df["HOURS_PER_DAY"] <= max(df["HOURS_PER_DAY"])), "NEW_HOURS_PER_DAY_STAGE"] = 3

    # Converting to numerical format
    # Define the mapping dictionary
    streaming_map = {"Spotify": 0, "YouTube Music": 1, "Apple Music": 2, "I do not use a streaming service.": 3,
                     "Pandora": 4, "Other streaming service": 4}

    # Use the map to replace the values, cast the column to int64 and drop the original column
    df["NEW_PRIMARY_STREAMING_SERVICE"] = df["PRIMARY_STREAMING_SERVICE"].replace(streaming_map).astype("int64")
    df.drop(["PRIMARY_STREAMING_SERVICE"], axis=1, inplace=True)

    # Define the mapping dictionary
    df["NEW_HPDxMH"] = df["HOURS_PER_DAY"] * df["MH_SCORE"]
    df["NEW_MH_OCD"] = (df["MH_SCORE"] - df["OCD"]) * df["OCD"]
    df["NEW_MH_SCORE"] = df["MH_SCORE"]
    df.drop(["MH_SCORE"], axis=1, inplace=True)

    df["NEW_COMPxINSTR"] = (df["COMPOSER"] + df["INSTRUMENTALIST"]).astype("int64")
    df["NEW_FOREIGNxEXPLORE"] = (df["FOREIGN_LANGUAGES"] * df["EXPLORATORY"]).astype("int64")
    df["NEW_FOREIGNxWORKING"] = (df["FOREIGN_LANGUAGES"] * df["WHILE_WORKING"]).astype("int64")

    # Splitting the time and binning into parts of the day
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df["TIME"] = df["TIMESTAMP"].dt.time

    df.loc[(df["TIME"] >= pd.to_datetime("00:00:00").time()) & (
            df["TIME"] < pd.to_datetime("06:00:00").time()), "NEW_DAY_PART"] = 1
    df.loc[(df["TIME"] >= pd.to_datetime("06:00:00").time()) & (
            df["TIME"] < pd.to_datetime("12:00:00").time()), "NEW_DAY_PART"] = 2
    df.loc[(df["TIME"] >= pd.to_datetime("12:00:00").time()) & (
            df["TIME"] < pd.to_datetime("18:00:00").time()), "NEW_DAY_PART"] = 3
    df.loc[(df["TIME"] >= pd.to_datetime("18:00:00").time()) & (
            df["TIME"] <= pd.to_datetime("23:59:59").time()), "NEW_DAY_PART"] = 4
    df["NEW_DAY_PART"] = df["NEW_DAY_PART"].astype("int64")
    df.drop(["TIMESTAMP"], axis=1, inplace=True)
    df.drop(["TIME"], axis=1, inplace=True)

    return df


df = feature_extraction(df)


# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, drop_first=False):
    categorical_cols = ["FAV_GENRE", "NEW_AGE_CLASS", "NEW_HOURS_PER_DAY_STAGE", "NEW_PRIMARY_STREAMING_SERVICE",
                        "NEW_DAY_PART"]
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, drop_first=True)

#Return the dataframe
def return_df():
    return df
