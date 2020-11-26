import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import xgboost as xgb


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#train_qs=pickle.load(open('train_qs.pkl', 'rb'))
stops=pickle.load(open('stops.pkl', 'rb'))
weights=pickle.load(open('weights.pkl', 'rb'))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    input_questions = ['what is your name', 'my name is azhar?']
    df_test = pd.DataFrame({'test_id': 0, 'question1': input_questions[0], 'question2': input_questions[1]}, index=[0])
    x_test = pd.DataFrame()
    x_test['word_match'] = df_test.apply(word_match_share, axis=1)
    x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1)
    d_test = xgb.DMatrix(x_test)
    p_test = model.predict(d_test)
    output=''
    if p_test[0]<0.01:
        output='Questions dont match :'
    else:
        output = 'Sorry Questions match :'
    return render_template('index.html', prediction_text=output+str(p_test[0]))

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=port)