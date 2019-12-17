# Sentence Order Based Sentiment Analysis 


• Proposed a sentiment classification model based on sentence order information
• Conducted extensive experiments on two real datasets in different languages, verified results against
classical models and drafted a report to summarize findings

## training phase

**Sentiment Classifier:**

We use training data to train a Sentiment Classifier $C_1$ to classify a Weibo (Twitter in China) text to be positive or negative

**Sentence Order Classifier:**

For one text, firstly we use Sentiment Classifier $C_1$ to predict whether it is positive or negative, predict label is $y$

Then, we split the text into $M$ sentences $[s_1, \cdots, s_M]$, for each sentence, we use Sentiment Classifier $C_1$ to predict whether it is positive or negative, if the sentiment of $s_j$ is the same as $y$, then $y^j=1$; otherwise, $y^j=0$, $j\in {1,2,\cdots,M}$, $y_j$ could be named as consistency label.

Then we use a Sentence Order Vector $S_j$ to represent every sentence, $S_j$ is a 10-dimension vector: 

X = [if it is the first sentence, if it is the second sentence, if it is the third sentence, if it is the fourth sentence, if it is the fifth sentence, if it is the third sentence from the end, if it is the fourth sentence from the end, if it is the third sentence from the end, if it is the second sentence from the end, if it is the last sentence]

y = $y^j$

Then we have training data for Sentence Order Classifier


## predicting phase


For a Weibo text, we first predict its sentiment by Sentiment Classifier $C_1$, and then split text into sentences. For each sentence, using $C_1$ predicts the sentiment of each sentence.

They use the sentence order classifier to predict the consistency label of each sentence, which can be understood as the reliability/confidence of the sentence.

And use this confidence to vote for sentiment of each sentence to get the final sentiment label

