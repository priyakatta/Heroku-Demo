# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:19:59 2020

@author: shvpr
"""


from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


power_app = Flask(__name__)


@power_app.route('/')
def home():
    return render_template('power_home.html')


@power_app.route('/predict', methods=['POST'])
def predict():
    xls = pd.ExcelFile(r"C:\Users\shvpr\Documents\Folds5x2_pp.xlsx")
    print(xls.sheet_names)

    df = pd.read_excel(r"C:\Users\shvpr\Documents\Folds5x2_pp.xlsx",
                       sheet_name='Sheet1')

    X = df.drop('PE', axis=1)
    y = df['PE']
    pipeline = Pipeline(steps=[
        ('std_scaler', StandardScaler()),
    ])
    df_prepared = pipeline.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        df_prepared, y, test_size=0.2, random_state=42)
    print(len(X_train), ' samples in training data\n',
          len(X_test), ' samples in test data\n', )
    param_grid = [

        {'bootstrap': [False], 'n_estimators':[10, 30],
         'max_features':[1, 2, 3, 4]}]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)
    if request.method == 'POST':
        AT = request.form['AT']
        AP = request.form['AP']
        Vacuum = request.form['Vacuum']
        RH = request.form['RH']
        data = [AT, AP, Vacuum, RH]
        vect = np.asarray(data, dtype='float32')
        vect = vect.reshape(1, -1)
        y_pred = grid_search.predict(vect)
        efficiency = round(((y_pred[0]/y.max())*100), 2)
    return render_template('result.html', prediction=efficiency)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080, use_reloader=False)
    power_app.run(debug=True, use_reloader=False)

# if __name__ == '__main__':
    # app.run(debug=True, use_reloader=False)
# $ export FLASK_APP=script2.py
# $ flask run --host 0.0.0.0 --port 5001
