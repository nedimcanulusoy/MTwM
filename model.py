import warnings
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from data_processing import return_df
import pickle

warnings.filterwarnings("ignore")

df = return_df()

# Model
def model_training():
    y = df["MUSIC_EFFECTS"]
    X = df.drop(["MUSIC_EFFECTS"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # 80 train, 20 test

    mtwm_model = LGBMClassifier(colsample_bytree=0.6, learning_rate=0.1, max_depth=2, n_estimators=50, subsample=0.6,
                                verbose=-1)
    mtwm_model.fit(X, y)
    return mtwm_model

# Make prediction based on user input
def predict(model, data):
    prediction = model.predict(data)
    return prediction

#Save model with pickle
def save_model(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

save_model(model_training())