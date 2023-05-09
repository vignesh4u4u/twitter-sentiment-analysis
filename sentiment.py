from flask import Flask , request , render_template
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle
import string
app = Flask(__name__,template_folder="template")
read = pickle.load(open("tweet.pkl","rb"))
@app.route("/")
def house():
    return render_template("sentiment.html")
@app.route("/pre",methods=["POST","GET"])
def predict():
    count = TfidfVectorizer()
    text = (request.form['text'])
    text = count.fit_transform([text])
    msginput = count.transform([text])
    predict = read.predict(msginput)[0]
    if (predict[0] == 0):  # [0]==1 means [msg]list have only one index value only that reason also given in [0]==1
        print('neutral tweet')
    elif (predict[0] == -1):
        print('negative tweet')
    elif (predict[0] == 1):
        print("positive tweet")
    else:
        print("enter text is false")
    return render_template("sentiment.html" , **locals())
if __name__ == "__main__":
    app.run(debug=True)

