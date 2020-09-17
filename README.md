<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/logo.png"></img>

# Using Elon's Tweets to Predict Tesla Stock Price 

## Introduction

**This project looked to use Elon's tweets to better predict Tesla stock price using Time Series.**

Financial price data was collected using the Yahoo Finance API

Tweet data was collected using the GetOldTweets3 API 

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



* [Traditional ML Classifications](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/classical_classification/elonVectorizerClassification.ipynb)

## BERT and DistilBERT

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/mhxx.gif" width="480"></img>

Quick Visualization of image data using a DCGAN requires minimal computing power.

       Generator uses Conv2DTranspose
       Discriminator uses Conv2D
       Hyperparameters:
          Filter
          kernel_size
          Stride
          Padding
          kernel_initializer

* [Elon BERT pytorch](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/elonBERTTorch/elonBERTtorch.ipynb)
* [Elon DistilBERT pytorch](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/distilBERT/elonDistilBERT.ipynb)


## XgBoost and Random Forest Regression 

Did a MinMaxScaler and took 1 difference for the sklearrn models and the tensorflow models. 

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

* [XgBoost and Random Forest Regression](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/trees/elonRFxgBoost.ipynb)

## RNN, LSTM, GRU 



Anime-face dataset was trained for 7 days using the default learning rate and mini batch repeat.  
Need to set the data path for the tfrecords to the location of your tfrecords.  

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/face_morph.gif" width="480"></img>

For MHXX dataset transfer learning was used and learning rate was adjusted to 0.001 and mini batch repeat to 1.
All adjustments were made inside the training/training_loop.py file.  You need to reference in the pickled model as well in resume_run_id and resume_kimg.    

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



