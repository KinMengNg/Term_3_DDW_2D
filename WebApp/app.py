from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle as pkl

from linear_regression_funcs import LinearRegression, r2_score, adjusted_r2_score, mean_squared_error


#Create and load the model
model = LinearRegression(saved_model='final_beta_means_stds.pkl')

#Create the flask app
app = Flask(__name__)

#Set start page
@app.route('/')
def root():
    return render_template("root.html")


#To convert the data into usable format and get the prediction
def ValuePredictor(to_predict_list):
    #to_predict needs to be a dataframe
    df_features_to_predict = pd.DataFrame(to_predict_list)
    df_features_to_predict = df_features_to_predict.transpose()

    df_features_to_predict.rename(columns={0:"Rice Produced (tonnes)", 1:"GDP per capita (constant 2015 US$)", 2:"Average Producer Price of Rice per kg (USD)", 3:"Population Size", 4:"Pump price for gasoline (US$ per liter)"}, inplace=True)

    prediction = model.predict_linreg(df_features_to_predict, model.beta, model.means, model.stds) 

    return prediction


#To send back data to the same html but with the predicted values this time
@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        #To grab values from html form, this is the easiest way I could think of to convert it to a list
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list)) #Convert to float
        #print(to_predict_list)

        result = round(float(ValuePredictor(to_predict_list)), 2)
        

        return render_template("root.html", result=result)


if __name__ == '__main__':
    app.run(debug=True)
