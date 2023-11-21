from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle as pkl

from linear_regression_funcs import predict_linreg, load_model

loaded_model = load_model('indo_beta_means_stds.pkl')
#print(loaded_model)

beta, means, stds = loaded_model


app = Flask(__name__)


@app.route('/')
def root():
    return render_template("root.html")


def ValuePredictor(to_predict_list):
    #to_predict needs to be a dataframe
    df_features_to_predict = pd.DataFrame(to_predict_list)
    df_features_to_predict = df_features_to_predict.transpose()

    df_features_to_predict.rename(columns={0:"Rice Produced (tonnes)", 1:"Inflation, consumer prices (annual %)", 2:"GDP per capita (constant 2015 US$)"}, inplace=True)

    prediction = predict_linreg(df_features_to_predict, beta, means, stds) 
    return prediction


@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list)) #Convert to float
        print(to_predict_list)

        result = round(float(ValuePredictor(to_predict_list)), 2)
        return render_template("root.html", result=result)


if __name__ == '__main__':
    app.run(debug=True)
