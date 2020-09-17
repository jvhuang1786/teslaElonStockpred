<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/logo.png"></img>

# Using Elon's Tweets to Predict Tesla Stock Price 

## Introduction

**This project looked to use Elon's tweets to better predict Tesla stock price using Time Series.**

* Financial price data was collected using the Yahoo Finance API

* Tweet data was collected using the GetOldTweets3 API 

* Data was collected from the period of Dec 1, 2011 to July 31, 2020

#### A webapp was also made for this project and can be found here:

#### [Tesla Elon Web APP](https://elon-tesla.herokuapp.com)

**Goal of the model was to see if we can use BERT to improve multivariate time series predictions**

### Libraries 

<table>

<tr>
  <td>Hypothesis</td>
  <td>scipy, statsmodel, plotly, numpy, pandas, matplotlib</td>
</tr>

<tr>
  <td>Data Wrangling</td>
  <td>GetOldTweet3, yahoofinance, re, vadersentiment, nltk, pandas_datareader, seaborn</td>
</tr>

<tr>
  <td>Data Cleaning</td>
  <td>pandas, sklearn, numpy, matplotlib</td>
</tr>

<tr>
  <td>Data Visualization</td>
  <td>itertools, collections, PIL, wordcloud, seaborn, nltk, bs4, bokeh, gensim, plotly</td>
</tr>

<tr>
  <td>Classicial Classification of Text</td>
  <td>seaborn, sklearn, matplotlib, xgboost</td>
</tr>

<tr>
  <td>BERT and DistilBert</td>
  <td>huggingface, ktrain, pytorch, tensorflow</td>
</tr>

<tr>
  <td>Arima</td>
  <td>scipy, statsmodel, math</td>
</tr>

<tr>
  <td>Xgboost and RandomForest regression</td>
  <td>sklearn, plotly, xgboost</td>
</tr>

<tr>
  <td>RNN, LSTM and GRU Neural Networks</td>
  <td>tensorflow, sklearn, statsmodel</td>

</table>


## Hypothesis 

Hypothesis using Kruskal- Wallis H-Test and Mann-Whitnney U Test:

To test if the distributions are different, I chose to do two tests. A Mann-Whitney U and a Kruskal-Wallis H test.

Observations were then split between three categories.

Business Tweets - 1352 tweet observations
Personal Tweets - 400 tweet observations
No Tweets - 928 tweet observations

     Null Hypothesis: the distributions of both samples are equal.

     Alternative Hypothesis: the distributions of both samples are not equal.
     
 The results are below: 

<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/Screen%20Shot%202020-09-16%20at%2011.07.16%20PM.png" width="480"></img>

Here is a boxplot of the different categories. 

<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/distribithypoth.gif" width="480"></img>

* [Hypothesis Notebook](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/notebooks/elonHypothesis.ipynb)

## Data Wrangling

For the Tweets:

	* There was about 10000 + tweets
	* They were collected using the GetOldTweet3 library from twitter.
	* Twitter handle was elonmusk
	* Tweet, fav_count and retweet_count was collected
	* To classify between business or personal a regex was used.  
	* I then read through all 10000 tweets to reclassify them.
	* Vader was used to create a label for positive or negative.
	
For Financial Data:
	
	* Yahoo data was collected using pandas_dataereader
	* Since Elon didn't tweet that much until December 1st, 2011 all financial data before that was not used.

Merging:
	
	* I then combined all the text data by day.  
	* Then matched the day of the tweet with its financial price.
	
* [Data Wrangling Notebook](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/notebooks/elon_wrangle.ipynb)



## Data Cleaning

Data Cleaning Code:

For text cleaning for the sklearn classification models the following code was used to do text cleaning. 

```python

def clean_tweet(tweet):
    #remove stopwords
    #use beautiful soup to remove the &/amps etc in tweets as well as website links
    soup_ = BeautifulSoup(tweet, 'lxml')
    soup_ = soup_.get_text()
    soup_ = re.sub(r'https?://[A-Za-z0-9./]+', '', soup_)

    #lowercase the words and remove punctuation
    lower_ = ''.join([word.lower() for word in soup_])

    #remove puncutations using a custom list
    punc_ = ''.join([punc(word) for word in lower_])
    #tokenize
    token_ = re.split('\W+',punc_)
    #remove stopwords
    stop_ = [word for word in token_ if word not in stopwords]
    tweet = ' '.join(word for word in stop_)

    return tweet
    
```

To fill in missing values for stock days an interpolate method was used.  When Looking at the data it kept missing values for finacial data the same.  For text data an empty string was used, and 0 for retweets, tweets and business or personal sentiment count. 


```python
#Features
X_train[stockDatax] = X_train[stockDatax].interpolate(method = 'linear', 
                                                    limit_direction="both")
X_test[stockDatax] = X_test[stockDatax].interpolate(method = 'linear', 
                                                    limit_direction="both")
```						   


* [Data Cleaning Notebook](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/notebooks/elon_clean.ipynb)


## Data Visualization

WordCloud, Bokeh count and word count plots were created.

A Word2Vec was used to show the relation among Elon's vocabulary.  

<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/Unknown-13" width="480"></img>


Also, the distribution of the tweets following in sentiment and type:

<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/distribution_classification.png" width="480"></img>



* [Data Visualization Notebook](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/notebooks/elon_visualization.ipynb)


## Classical Text Classification 

Sklearn library was used to see if we could classify the text into the 6 categories first.  However, the results were not great even with hypertuning. 

The best model after hypertuning had a f1-score of 0.62.  Below are some of the results:

<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/Screen%20Shot%202020-09-17%20at%2012.00.09%20AM.png" width="480"></img>

* [Traditional ML Classifications](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/classical_classification/elonVectorizerClassification.ipynb)

## BERT and DistilBERT

Using a BERT and DistilBERT yielded much better results.  The BERT model f1-score was 0.80 around and DistilBERT was about 0.79

80 tokens were chosen given the distribution. 

<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/f2859e575638aacebdaa56cce1ae2862f13b743c462e97320fe05116.jpeg-2.png" width="480"></img>

Below is the confusion matrix for DistilBERT

<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/confusion_matrix_distil.png" width="480"></img>

DistilBERT model was chosen since it was significantly smaller at 260 mb vs 1.4 gb(BERT). 

* [Elon BERT pytorch](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/elonBERTTorch/elonBERTtorch.ipynb)
* [Elon DistilBERT pytorch](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/distilBERT/elonDistilBERT.ipynb)


## XgBoost and Random Forest Regression 

I tried using classical ml models to do a time series prediction.  However, because of the structure of these algorithms they wouldn't be able be a good predictor if the stock kept changing at a rapid price. 

First we tested if it was stationary by using the Dickey Fuller test.  It wasn't so I took the first difference and tested it again and it was. 

To inverse the scaler and difference the folowing code was used. 

```python
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

Feature importance after hyperparameter tuning on xgBoost:

```python
retweet_count 0.0014913935
fav_count 0.0011018803
tweetLen 0.002772917
Business positive 0.0014061782
Business neutral 0.0023037135
Business negative 0.0
Personal positive 0.0037213876
Personal neutral 0.0014253978
Personal negative 0.004577721
Open 0.9800373
Volume 0.0011621305
```

* [XgBoost and Random Forest Regression](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/trees/elonRFxgBoost.ipynb)

## RNN, LSTM, GRU 

I tried 3 neural networks.  GRU produced overall the best results. 

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/face_morph.gif" width="480"></img>

Final GRU architecture is below: 

```python
#Learning Rate
lr = 0.0001
#Set RandomSeed
tf.random.set_seed(777)

#Reshape Structure (number of observations, time steps, number of features)

# fit an GRU network to training data
def fit_gru(train, test, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    X_t, y_t = test[:, 0:-1], test[:, -1]
    X_t = X_t.reshape(X_t.shape[0], 1, X_t.shape[1])
    model = Sequential()
    model.add(GRU(neurons, return_sequences = True, recurrent_activation="relu", activation = 'elu',
              input_shape = (X.shape[1], X.shape[2]),dropout = 0.3))
    model.add(Dense(1))
    adam = optimizers.Adam(lr= lr, amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # Setting up an early stop
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50,  verbose=2, mode='min')
    callbacks_list = [earlystop]
    model.fit(X,y, epochs = nb_epoch, batch_size = batch_size, verbose = 2, shuffle = False, validation_data = (X_t, y_t),
             callbacks=callbacks_list)
    return model

#Batch Size: 8
#Epochs: 1000
#Neurons: 500

gru_model = fit_gru(train_scaled, test_scaled, 8, 1000, 500)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, :-1].reshape(len(train_scaled), 1, 11)
history = gru_model.predict(train_reshaped)

```

Predictions using DistilBERT to classify first and then add it in the dataframe for the GRU timeseries prediction.

Before PyBay2020 predictions were made from August 1st, 2020 to August 13th, 2020.  Using the saved scaler on the training data as well as the saved GRU model.

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/armor_morph.gif" width="480"></img>

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/armor_morph.gif" width="480"></img>

* [RNN, LSTM, GRU Notebook](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/lstm_gru/elon_lstm_gruv1.ipynb)

## Author

* Justin Huang

## Resources

* [Machine Learning Mastery LSTM Neural Networks](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

* [Time Series using Arima](https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/)

* [Using BERT in PyTorch](https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)

* [What are RNNs](https://www.youtube.com/watch?v=LHXXI4-IEns)

* [What is BERT](http://jalammar.github.io/illustrated-bert/)



