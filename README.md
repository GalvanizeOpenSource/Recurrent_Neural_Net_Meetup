# Recurrent Neural Networks for Stock Price Prediction

At the Meetup, we'll be using Python 3 and `numpy` for about 2/3 of the evening.  The last third will use Keras.  You can use either the Theano or Tensorflow backend.
We'll be working out of a text editor and the IPython console.  Starter scripts are found in this repo, and they are written in such a way that you could put blocks in Jupyter Notebooks and evaluate them there instead.

## Keras install instructions
* Install keras using the command `pip install keras`

* In the IPython shell, type `import keras`. It will fail (it can't find the `tensorflow` backend), but the whole point is to create the file at `~/.keras/keras.json` that you need to edit to use the `theano` backend if you are using Theano. Read about that [here.](https://keras.io/backend/)

This talk is based on resources from [iamtrask](http://iamtrask.github.io/), [Andrey Karpathy](http://cs231n.stanford.edu/2016/syllabus), and [Jakob Aungiers.](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction)  Many of the scripts in this repo originated from their blogs, repos, and Github pages and you'll find a more detailed citation in the script.  Most scripts have been modified - all errors induced (or fixed!) are my own.

