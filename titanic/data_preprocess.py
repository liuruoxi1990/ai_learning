# -*- coding: utf-8 -*



import os
import sys
import pandas as pd
import numpy as np

import sklearn.preprocessing as preprocessing
from sklearn import cross_validation


from sklearn import linear_model


TRAIN_DATASETS_PATH="/home/lrx/dataset/train.csv"
TEST_DATASETS_PATH="/home/lrx/dataset/test.csv"

pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)

def train_prep():

    train_data = pd.read_csv(TRAIN_DATASETS_PATH)

    train_data.loc[(train_data.Cabin.notnull()), "Cabin"] = "Yes"
    train_data.loc[(train_data.Cabin.isnull()), "Cabin"] = "No"

    train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    train_data["Fare"].fillna(train_data["Fare"].mean(), inplace=True)



    dummies_Cabin = pd.get_dummies((train_data["Cabin"]), prefix="Cabin")
    dummies_Embarked = pd.get_dummies((train_data["Embarked"]), prefix="Embarked")
    dummies_Sex = pd.get_dummies((train_data["Sex"]), prefix="Sex")
    dummies_Pclass = pd.get_dummies((train_data["Pclass"]), prefix="Pclass")

    df = pd.concat([train_data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    #import pdb;
    #pdb.set_trace()
    scaler = preprocessing.StandardScaler()
    age_scale = scaler.fit(train_data["Age"].values.reshape(-1, 1))
    fare_scale = scaler.fit(train_data["Fare"].values.reshape(-1, 1))

    df["Age_scale"] = scaler.fit_transform(train_data["Age"].values.reshape(-1, 1), age_scale)
    df["Fare_scale"] = scaler.fit_transform(train_data["Fare"].values.reshape(-1, 1), fare_scale)

    return df


def test_prep():
    test_data = pd.read_csv(TEST_DATASETS_PATH)

    test_data.loc[(test_data.Cabin.notnull()), "Cabin"] = "Yes"
    test_data.loc[(test_data.Cabin.isnull()), "Cabin"] = "No"

    test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)
    test_data["Fare"].fillna(test_data["Fare"].mean(), inplace=True)

    dummies_Cabin = pd.get_dummies((test_data["Cabin"]), prefix="Cabin")
    dummies_Embarked = pd.get_dummies((test_data["Embarked"]), prefix="Embarked")
    dummies_Sex = pd.get_dummies((test_data["Sex"]), prefix="Sex")
    dummies_Pclass = pd.get_dummies((test_data["Pclass"]), prefix="Pclass")

    df = pd.concat([test_data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    # import pdb;
    # pdb.set_trace()
    scaler = preprocessing.StandardScaler()
    age_scale = scaler.fit(test_data["Age"].values.reshape(-1, 1))
    fare_scale = scaler.fit(test_data["Fare"].values.reshape(-1, 1))

    df["Age_scale"] = scaler.fit_transform(test_data["Age"].values.reshape(-1, 1), age_scale)
    df["Fare_scale"] = scaler.fit_transform(test_data["Fare"].values.reshape(-1, 1), fare_scale)

    return df

def train(train_d, test_d):
    train_np  = train_d.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').values
    train_df = train_d.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

    y = train_np[:, 0]
    x = train_np[:, 1:]

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(x, y)
    print cross_validation.cross_val_score(clf, x, y, cv=5)

    print clf

    test_np = test_d.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test_np)
    result = pd.DataFrame(
        {'PassengerId': test_d['PassengerId'].values, 'Survived': predictions.astype(np.int32)})

    #result.to_csv("/tmp/logistic_regression_predictions.csv", index=False)

    import pdb; pdb.set_trace()
    print pd.DataFrame({"column_name":list(train_df.columns)[1:], "coef": list(clf.coef_.T)})



def main():
    train_d = train_prep()
    test_d = test_prep()
    train(train_d, test_d)



if __name__ == "__main__":
    main()