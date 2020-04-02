import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
import matplotlib.pyplot as plt

EPOCHS = 200

# class for callback function that stop training model after  reached target metric
class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.90):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

# display accuracy  and loss while training
def displayModelHistory(model_history):
    graph_history = model_history
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('Training - Accuracy and Loss', fontsize=12)

    plt.subplot(121)
    line1, = plt.plot(range(1, len(graph_history['acc']) + 1), graph_history['acc'], label='training')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(handles=[line1])
    plt.grid(True)

    plt.subplot(122)
    line1, = plt.plot(range(1, len(graph_history['loss']) + 1), graph_history['loss'], label='training')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(handles=[line1])
    plt.grid(True)


print(tf.__version__)

# load data
titanic = pd.read_csv("train_Titanic.csv")
titanic_test = pd.read_csv("test_Titanic.csv")


# features clean and fill
# fill out Age column with median value of age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

# Replace all the occurrences of male with the number 0 and female with 1.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')
# convert to digit for Embarked
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].mean())


titanic["SibSp"] = titanic["SibSp"].fillna(0)
titanic_test["SibSp"] = titanic_test["SibSp"].fillna(0)

titanic["Parch"] = titanic["Parch"].fillna(0)
titanic_test["Parch"] = titanic_test["Parch"].fillna(0)

titanic["Relatives"] = titanic["Parch"] * titanic["SibSp"]
titanic_test["Relatives"] = titanic["Parch"] * titanic["SibSp"]

print(titanic.info())

# choose predictors
predictors = ['Pclass', 'Sex', 'Fare', 'Age', 'Embarked', 'Relatives']

# scale 'Fare', 'Age'
titanic["Fare"] = titanic["Fare"] / titanic["Fare"].max()
titanic["Age"] = titanic["Age"] / titanic["Age"].max()


# init tha callback
callbacks = MyCallback()

# create model
model = keras.Sequential([
    keras.layers.Dense(6, activation='sigmoid', input_shape=[6, ]),
    keras.layers.Dense(8, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])

# set parameters for training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# learn !
history = model.fit(titanic[predictors].to_numpy(),  titanic["Survived"].to_numpy(), epochs=EPOCHS, callbacks=[callbacks])

displayModelHistory(history.history)

######################################### K fold evaluation #############################

eval = []
n_split = 4
X = titanic[predictors].to_numpy()
Y = titanic["Survived"]
for train_index, test_index in KFold(n_split).split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model.fit(x_train, y_train, epochs=EPOCHS,
                        callbacks=[callbacks])
    eval.append(model.evaluate(x_test, y_test))
print("All evaluations: ", str(eval))

dfObj = pd.DataFrame(eval)
dfObj.columns = ['Loss', 'Acc']
print("Average loss - ", dfObj["Loss"].mean(), "Average accuracy - ", dfObj["Acc"].mean())

###########################################################################################

# predict
test_predict = model.predict(titanic_test[predictors].to_numpy())
test_predict[test_predict <= .5] = 0
test_predict[test_predict > .5] = 1
test_predict = test_predict.astype(int)
test_predict = np.reshape(test_predict, len(test_predict))

# submit result
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": test_predict
    })

submission.to_csv('titanic.csv', index=False)



