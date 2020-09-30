<!-- necessary to render the tables nicely on the CN blog -->
<style>
  table-div {
    margin-bottom: 20px;
  }
  table, th, td {
    padding: 10px;
    text-align: center;
    border: 1px solid gray;
  }
  th {
    color: #545454;
    background-color: #ededed;
  }
</style>
<strong>By Cagdas Yetkin, Data Scientist at Nokia</strong>

One of my favorite things about working with data is exploring datasets that everyone can relate to. Although many might strongly disagree on which genre is the best, I bet most of us can agree that we enjoy watching a great movie!

In this data analysis example, you will analyze a dataset of movie ratings to draw various conclusions. You will learn how to:

1. Get and Clean Data
2. Understand and interpret the overall figures and basic statistics
3. Join datasets, and aggregate and filter your data by conditions
4. Discover hidden patterns and insights
5. Create summary tables

This tutorial teaches you to perform all of the above tasks using Python and its popular <a href="https://pandas.pydata.org/docs/getting_started/index.html#getting-started" target="_blank"><code>pandas</code></a> and <a href="https://matplotlib.org/3.2.2/contents.html" target="_blank"><code>matplotlib</code></a> libraries. You can download and run the Jupyter Notebook used in this data analysis example <a href='https://github.com/CodingNomads/articles/blob/main/movie-tweets/movie-analysis.ipynb' target='_blank'><strong>here</strong></a>.

<strong>Table of Contents</strong>

- [Introduction](#introduction)
- [Inspect the Data](#inspect-the-data)
- [Set Up Your Notebook](#set-up-your-notebook)
- [Read in the Data](#read-in-the-data)
  - [Users: users.dat](#users-usersdat)
  - [Ratings: ratings.dat](#ratings-ratingsdat)
  - [Movies: movies.dat](#movies-moviesdat)
- [Explore Your Data](#explore-your-data)
- [Join the Datasets](#join-the-datasets)
- [Visualize Patterns](#visualize-patterns)
- [Explore a Question](#explore-a-question)
  - [Top Rated Sci-Fi Movies by Decades](#top-rated-sci-fi-movies-by-decades)
- [What Next?](#what-next)

<h2 id="introduction">Introduction: Movie Ratings Data Analysis Example</h2>

You can download the data from the original GitHub repo - <a href="https://github.com/sidooms/MovieTweetings" target="_blank">Movie Tweetings Project</a>.

The data in this example consists of  movie ratings from Twitter since 2013, updated daily. The data was created from people who connected their <a href='https://www.imdb.com/' target='_blank'>IMDB</a> profile with their <a href='https://twitter.com/' target='_blank'>Twitter</a> accounts. Whenever they rated a movie on the IMDB website, an automated process generated a standard, well-structured tweet.

These _well-structured_ tweets look like this:

> "I rated The Matrix 9/10 http://www.imdb.com/title/tt0133093/ #IMDb"

Because of this nice structure, we can use this data to learn and practice data analysis using Python.

<strong>Tip:</strong> You are highly encouraged to write the code for this data analysis example yourself! This will help you truly understand the contents of this tutorial, give you the practice you need to improve your data analysis "muscle memory" skills, and you may discover some additional interesting revelations for yourself!

<h2 id="inspect-the-data">Inspect the Data</h2>

To get started, confirm that you have these 3 files in your working directory:

- `users.dat`
- `movies.dat`
- `ratings.dat`

If all these files are accessible to you, you can start off your investigation by checking what these files contain. Let's start off by looking at the first three lines in `users.dat` directly in your terminal:

```bash
head -n3 data/users.dat
```
&nbsp;

Your output will look similar to this:

```text
1::139564917
2::522540374
3::475571186
```
&nbsp;

At first it may be confusing that you can't see any field names but these are documented in the <a href="https://github.com/sidooms/MovieTweetings" target="_blank">README</a> file as follows:

> In `users.dat` the _first_ field is the `user_id` and the _second_ one is `twitter_id`.

You can see that there is a surprising amount of colons in this data snippet. Because you already know that you are working with two data fields, this means that the creators of this dataset decided to use a double-colon `::` as a field separator. Interesting choice! It is helpful to keep in mind that data fields can be divided by all sorts of different separators, and it's good to know which one is used in the data you are working with.

With a basic idea of what you can expect to see in `users.dat`, let's next take a peek into `movies.dat`:

```bash
head -n3 data/movies.dat
```
&nbsp;

The output of this file will look like this:

```text
0000008::Edison Kinetoscopic Record of a Sneeze (1894)::Documentary|Short
0000010::La sortie des usines Lumière (1895)::Documentary|Short
0000012::The Arrival of a Train (1896)::Documentary|Short
```
&nbsp;

In this file, you have three fields:

1. `movie_id`
2. `movie_title`
3. `genres`

A single movie can belong to more than one genre, and the `genres` are separated by pipe characters `|`, another interesting choice!

After looking at `movies.dat`, there's only one file left to inspect. Let's peek into `ratings.dat` next:

```bash
head -n3 data/ratings.dat
```
&nbsp;

The output you will receive should look similar to the one below:

```text
1::0114508::8::1381006850
2::0102926::9::1590148016
2::0208092::5::1586466072
```
&nbsp;

In this third dataset, your variables are:

1. `user_id`
2. `movie_id`
3. `rating` and
4. `rating_timestamp`

And again it comes with an interesting feature: The timestamps are in <a href="https://www.unixtimestamp.com/" target="_blank">unixtime</a> format!

**UNIX time** is a time format often used in computer time that shows the seconds passed since January 1st, 1970. You can use online converters to translate it to a format that is easier to read for humans. If you're interested, read more about <a href='https://en.wikipedia.org/wiki/Unix_time' target='_blank'>Unix time on Wikipedia</a>. 

<div style="width:100%;height:0;padding-bottom:50%;position:relative;"><iframe src="https://giphy.com/embed/9u514UZd57mRhnBCEk" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/reaction-9u514UZd57mRhnBCEk">via GIPHY</a></p></div>
&nbsp;

<h2 id="set-up-your-notebook">Set Up Your Notebook</h2>

Now you have an overall understanding of how the raw datasets look. Next, you will import the libraries you will need for the rest of this analysis:

```python
import warnings

import pandas as pd
import numpy as np
import scipy as sc

import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('fivethirtyeight')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')
```

Let's look a bit closer at the options you set up in the code snippet above. You:

- Give the `filter-out-warnings` command to have a cleaner notebook without warning messages.
- Set the max rows and max columns to some big numbers, in this case `50`. This option just makes all the columns and rows in a DataFrame more readable or visible.
- Use `fivethirtyeight` style to have plots like the ones on <a href="https://www.fivethirtyeight.com" target="_blank">fivethirtyeight.com</a>: a website founded by <a href="https://en.wikipedia.org/wiki/Nate_Silver" target="_blank">Nate Silver</a>. If you want to explore `fivethirtyeight` further, I highly recommend the book: <a href="https://www.amazon.com/Signal-Noise-Many-Predictions-Fail-but/dp/0143125087" target="_blank">The Signal and the Noise</a>.

These imports and adjustments create a good base setup for you to get started with your analysis. Keep in mind that while the `import`s are necessary, the adjustments are just to make your analysis easier and better-looking.

<h2 id="read-in-the-data">Read in the Data</h2>

After importing the necessary libraries, you are now ready to read the files into `pandas` data frames.

There are a couple of adjustments you should make while reading in the data, to make sure it will be in good shape to work with:

- Define that the separators are double colons `::`
- Give the column names, so they will become the headers
- Convert the UNIX time to a datetime format

With this in mind, let's read in `users.dat`, `ratings.dat` and `movies.dat` one by one:

<h3 id="users-usersdat">Users: `users.dat`</h3>

Starting with `users.dat`, the following code snippet will read in the file into your notebook, register the double-colon as the separator between the fields, and add column names as well:

```python
users = pd.read_csv('data/users.dat', sep='::',
                    names=['user_id', 'twitter_id'])
```

This creates a `DataFrame()` object, and you can check the first few entries of this table-like object with the `.head()` method:

```python
users.head()
```

You will see a nicely formatted output that shows the first 5 rows of your `users` data frame:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>twitter_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>139564917</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>522540374</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>475571186</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>215022153</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>349681331</td>
    </tr>
  </tbody>
</table>
</div>

You successfully read in the data from the external file and now have access to it as a `DataFrame()` object. Let's do the same with the other files as well.

<h3 id="ratings-ratingsdat">Ratings: `ratings.dat`</h3>

Similar to before, you will want to read in the data and save it into a data frame, define the separator, and pass in the column names. Additionally, you will also call the `.sort_values()` method on the data frame right away, to sort your data by when the ratings have been created:

```python
ratings = pd.read_csv('data/ratings.dat', sep='::',
                      names=['user_id', 'movie_id', 'rating', 'rating_timestamp']
                      ).sort_values("rating_timestamp") # sorting the dataframe by datetime
```

You will also want to convert the `rating_timestamp` values to actual `datetime` format, and you can do that in `pandas` like so:

```python
ratings["rating_timestamp"] = pd.to_datetime(ratings["rating_timestamp"], unit='s')
```

Let's peek into the first 5 rows of your newly created `ratings` data frame:

```python
ratings.head()
```

You output should look similar to below:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>rating_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138461</th>
      <td>11080</td>
      <td>2171847</td>
      <td>6</td>
      <td>2013-02-28 14:38:27</td>
    </tr>
    <tr>
      <th>585269</th>
      <td>45890</td>
      <td>444778</td>
      <td>8</td>
      <td>2013-02-28 14:43:44</td>
    </tr>
    <tr>
      <th>611517</th>
      <td>47821</td>
      <td>1411238</td>
      <td>6</td>
      <td>2013-02-28 14:47:18</td>
    </tr>
    <tr>
      <th>648464</th>
      <td>50454</td>
      <td>1496422</td>
      <td>7</td>
      <td>2013-02-28 14:58:23</td>
    </tr>
    <tr>
      <th>742847</th>
      <td>58297</td>
      <td>118799</td>
      <td>5</td>
      <td>2013-02-28 15:00:53</td>
    </tr>
  </tbody>
</table>
</div>

With the `ratings` data read in, there's only one more file left to go.

<h3 id="movies-moviesdat">Movies: `movies.dat`</h3>

Of course you also need to have access to information about the actual movies, to find potential correlations e.g. between ratings and movie genres. So, let's read in that data next:

```python
movies = pd.read_csv('data/movies.dat', sep='::',
                     header=None, names=['movie_id', 'movie_title', 'genres'])
```

Checking the successful completion of this process with the familiar `movies.head()` command, you will see something similar to below:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>movie_title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>Edison Kinetoscopic Record of a Sneeze (1894)</td>
      <td>Documentary|Short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>La sortie des usines Lumière (1895)</td>
      <td>Documentary|Short</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>The Arrival of a Train (1896)</td>
      <td>Documentary|Short</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>The Oxford and Cambridge University Boat Race ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>91</td>
      <td>Le manoir du diable (1896)</td>
      <td>Short|Horror</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

With this, the data has been read in to the notebook. What follows next, is **exploration**.

<div style="width:100%;height:0;padding-bottom:56%;position:relative;"><iframe src="https://giphy.com/embed/l49K0XNJvLtx9lPry" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/l49K0XNJvLtx9lPry">via GIPHY</a></p></div>
&nbsp;

<h2 id="explore-your-data">Explore Your Data</h2>

To get a feeling for the data you are working with, it always helps to play around a little and create some quick stats and graphs for different aspects of it. This will help you have a better overview of what the data is about.

Since you want to find out how well movies are liked or disliked, the most important variable is the movie `rating`. Let's see its distribution:

```python
ratings['rating'].value_counts()
```

Your output should look similar to the one you can see below:

```text
8     211699
7     196410
9     124459
6     114372
10    103648
5      65907
4      26940
3      14759
1      10324
2       8778
0        267
Name: rating, dtype: int64
```

`value_counts()` is a quick but effective way of checking what values your variable takes. Here we see quickly that the rating score `8` was given `211699` times!

Let's keep exploring. A **histogram** will show you the distribution and the `describe()` method will give additional **basic statistics**. Both of them are quite helpful to get quick insights, so let's try them out next:

```python
ratings['rating'].describe()
```

As mentioned, the `.describe()` method will display basic statistics about a column, so here they are for the `rating` column:

```text
count    877563.000000
mean          7.316577
std           1.853619
min           0.000000
25%           6.000000
50%           8.000000
75%           9.000000
max          10.000000
Name: rating, dtype: float64
```

Next, let's look at a visual representation of the data by creating a histogram:

```python
ratings['rating'].hist(bins=10)
```

The data with the above settings will produce a histogram that looks like this:

![Rating Histogram with Bin of 10](https://github.com/CodingNomads/articles/blob/main/movie-tweets/imgs/output_30_0.png?raw=true)

You'll noticed that it is skewed to the left! That means that the distribution doesn't have a symmetrical shape around the mean, and this specific off-balanced distribution has a long tail on the left hand side.

The `hist()` and `describe()` methods are in fact quite similar: One gives text output and the other gives its visual representation.

Given that both functions return the same output, you may also be able to conclude that the `rating` is left-skewed by looking only at the text output of your `.describe()` method. The relevant data for this conclusion are:

- The `mean` is much smaller than the `median` and
- 25% of the data covers only until a rating of `6`

This is a bit confusing. You have seen first that the highest frequency was `8`. And then, after generating the histogram, it looked like the ratings were highest around `9`-`10`.

This difference can arise because of <a href="https://statistics.laerd.com/statistical-guides/understanding-histograms.php" target="_blank">binning</a>. Different amounts of bins will lead to different results. Most of the time, the person conducting the analysis decides the right number of bins after a few trials. Generally, you will have a better idea about what is the right bin size for your data set after some research and digging into it.

Playing with the bins of a histogram can have an impact on the story you are telling. The same histogram would look like this if you increase the number of bins from `10` to `30`:

```python
ratings['rating'].hist(bins=30)
```

![Rating Histogram with Bin of 30](https://github.com/CodingNomads/articles/blob/main/movie-tweets/imgs/output_35_0.png?raw=true)

You can see that this can lead to a different conclusion. If you were using the _first_ histogram you would falsely argue that the most frequent rating was `9` or maybe `10`. However, the _second_ one makes everything crystal clear and shows that the most frequent rating lies at `8` instead. Also, note that if you use the `.value_counts()` method, you wouldn't fall into that trap.

Thanks to these methods now you have a more clear understanding about the `rating` variable in your data. You will focus on the `user_id` column next.

How many unique `user_id` do you have in the `users` data?

```python
f"You have {len(users.user_id.unique())} unique user ids in the data"
```

> 'You have 68388 unique user ids in the data'

You have seen earlier that both `value_counts()` and `describe()` are quite handy. So why not combine them to learn a little more?

For instance, how many rating tweets are posted by a user on average? What is the minimum, maximum and median number of tweets posted by the users? The answer to these questions will enable you understand how active the users are: Are they frequent users or are they disappearing after shooting one single tweet? 

Let's try it out:

```python
ratings.user_id.value_counts().describe()
```

Running the code snippet above, you will receive another block of text-based statistics as your output:

```text
count    68388.000000
mean        12.832120
std         46.009589
min          1.000000
25%          1.000000
50%          2.000000
75%          7.000000
max       2875.000000
Name: user_id, dtype: float64
```
&nbsp;

Notice that this time you accessed the column using **dot notation**. In this case it does the same as accessing it through the square-bracket notation you used before, but is a little bit more convenient. Check out <a href='https://stackoverflow.com/a/55057329' target='_blank'>this StackOverflow post</a> if you want to learn more about the limitations and differences between the two notations.

See in the above output how the `mean` is much greater than the `median` (12.83 vs 2). It means that the data is skewed to the right.

This skewness is at the extreme: Look how the `max` value is far, far away! Could there be someone posting more than 2000 times? Not likely.

The output also tells us that _50%_ of the people used it only _twice_ but the `mean` is almost `13`. This is because of those users with extremely high usage numbers. 

Could it be possible that they are not human beings but bots instead? That could be a great investigation topic, if you want to dive deeper. 

But for this data analysis example, let's leave this aside for now and continue by <strong>joining the datasets</strong> we have.

<h2 id="join-the-datasets">Join the Datasets</h2>

Joining data could be really difficult, as this tweet addresses:

![Joining before Pandas Twitter](https://github.com/CodingNomads/articles/blob/main/movie-tweets/imgs/tweet_pandas.PNG?raw=true)

Luckily, with `pandas` you have a user-friendly interface to <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html" target="_blank">join</a> your `movies` data frame with the `ratings` data frame. This is going to be an _inner_ join. It means that you are bringing in the movies only if there is a rating available for them:

```python
movies_rating = (ratings
                  .set_index("movie_id")
                  .join(movies.set_index("movie_id"),
                        how="left")
                 )

movies_rating.head(2)
```
&nbsp;

Inspecting the first two rows with the `.head(2)` method shows you this:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>rating</th>
      <th>rating_timestamp</th>
      <th>movie_title</th>
      <th>genres</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>41412</td>
      <td>5</td>
      <td>2014-04-08 18:20:11</td>
      <td>Edison Kinetoscopic Record of a Sneeze (1894)</td>
      <td>Documentary|Short</td>
    </tr>
    <tr>
      <th>10</th>
      <td>68190</td>
      <td>10</td>
      <td>2014-10-09 18:15:53</td>
      <td>La sortie des usines Lumière (1895)</td>
      <td>Documentary|Short</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

Notice that you didn't use the `on` and `how` parameters when you joined the data, because you set the index of both data frames to `movie_id`. So, the `.join()` method knew on which variable to join and by default this creates an _inner_ join.

Looking at the output of the `.join()` operation, you have a new problem: You want to quantify the _genres_, but how would you count them?

One way of doing that could be creating dummies for each possible `genre`, such as _Sci-Fi_ or _Drama_, and having a single column for each. Creating dummies means creating `0`s and `1`s just like you can see in the example below:

```python
dummies = movies_rating['genres'].str.get_dummies()
dummies.head()
```
&nbsp;

The data frame that gets produced by this command looks like this:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Action</th>
      <th>Adult</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>Game-Show</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>News</th>
      <th>Reality-TV</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Talk-Show</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

You can concatenate these `dummies` to the original `movies_rating` data frame:

```python
tidy_movie_ratings = (pd.concat([movies_rating, dummies], axis=1)
                       .drop(["rating_timestamp", "genres"], axis=1)
                )

tidy_movie_ratings.head()
```
&nbsp;

Your newly created data frame will look like this:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>rating</th>
      <th>movie_title</th>
      <th>Action</th>
      <th>Adult</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>Game-Show</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>News</th>
      <th>Reality-TV</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Talk-Show</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>41412</td>
      <td>5</td>
      <td>Edison Kinetoscopic Record of a Sneeze (1894)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>68190</td>
      <td>10</td>
      <td>La sortie des usines Lumière (1895)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>67178</td>
      <td>10</td>
      <td>The Arrival of a Train (1896)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>36321</td>
      <td>8</td>
      <td>The Oxford and Cambridge University Boat Race ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>5608</td>
      <td>6</td>
      <td>Le manoir du diable (1896)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

This is almost as tidy as you want it, but it would be much more clean and useful if you could get those production years in a separate column. That would allow you to compare film productions over the years.

To accomplish this, you will practice working with the `.str` attribute, which is quite popular - and a lifesaver in many cases! You will:

- Make a new column by getting the 4 digits representing the year
- Remove the last 7 characters from the movie names
- Checkout the result

Let's write the code for achieving these tasks:

```python
tidy_movie_ratings["production_year"] = tidy_movie_ratings["movie_title"].str[-5:-1]
tidy_movie_ratings["movie_title"] = tidy_movie_ratings["movie_title"].str[:-7]
```
&nbsp;

Before checking out the results, let's go ahead and reset the index on this data frame first:

```python
tidy_movie_ratings.reset_index(inplace=True)

tidy_movie_ratings.head(2)
```
&nbsp;

Now you can see that you produce a better-formatted version of the data frame:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>user_id</th>
      <th>rating</th>
      <th>movie_title</th>
      <th>Action</th>
      <th>Adult</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>Game-Show</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>News</th>
      <th>Reality-TV</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Talk-Show</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
      <th>production_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>41412</td>
      <td>5</td>
      <td>Edison Kinetoscopic Record of a Sneeze</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1894</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>68190</td>
      <td>10</td>
      <td>La sortie des usines Lumière</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1895</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

**Congratulations!** With this, you have completed the most difficult part of this data analysis example: Getting and cleaning the data. Let's quickly recap what you did so far:

- You read the raw data into data frames
- You learned and reported basic statistics
- You joined data frames and created new fields

You did some great work if you followed all the way until here! You can now: <a href="https://www.youtube.com/watch?v=2wnOpDWSbyw" target="_blank">watch the first movie in your records from 1894</a> as a reward :)

Next, you are going to visualize your data and discover some patterns. When delivering a report in a professional or academic setting, this is where things start to get very interesting!

<div style="width:100%;height:0;padding-bottom:40%;position:relative;"><iframe src="https://giphy.com/embed/VeNDat4n4Kre76oS1g" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/UpSteamMobileCarWash-data-eesti-upsteamers-VeNDat4n4Kre76oS1g">via GIPHY</a></p></div>
&nbsp;

<h2 id="visualize-patterns">Visualize Patterns</h2>

First, you will start with visualizing the total volume of films created over the years.

Next, you will count the total number of productions for each year and plot it. The record you see for the year of 2021 should be filtered out before proceeding:

```python
condition = tidy_movie_ratings["production_year"].astype(int) < 2021

prodcount = (tidy_movie_ratings[condition][["production_year", "movie_id"]]
             .groupby("production_year")
             .count()
            )

prodcount.tail()
```

Similar to the `.head()` method you have encountered before, `.tail()` shows you a subset of the rows of your data frame. However, instead of showing the _first_ ones, it shows you the _last_ ones:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
    </tr>
    <tr>
      <th>production_year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016</th>
      <td>80425</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>62035</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>43694</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>50044</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>5712</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

Aside from 2021, which you filtered out, the other interesting year here is 2020. Although more than half of the year 2020 has passed at the time of writing this article, there are only `5712` rated films and movies for the year so far. Looks like 2020 is one of the most extraordinary years in history? Or maybe the movies are so new, that people didn't have the time to watch them yet. Or both!

You can chart a 5 year moving average of the total productions:

```python
(prodcount
 .rolling(5).mean().rename(columns={"movie_id":"count"})
 .plot(figsize=(15,5),
       title="Count of Rated Movies - by production year")
)
```
&nbsp;

This will produce a graphic similar to the one below:

![5-year moving average plot](https://github.com/CodingNomads/articles/blob/main/movie-tweets/imgs/output_63_0.png?raw=true)

You can see that the 5-year moving average is in a shocking decline! What is happening here? What can be the reason? Can you formulate some hypotheses? Here are some points for you to consider:

- This was an _inner_ join. So these are the _rated_ movies. Perhaps site and app usage went down.
- The filming industry is in a serious crisis! They are not producing films because of COVID-19.
- People didn't have time to watch the most recent movies. If they didn't watch them, they don't rate them, and you can see a decline in ratings. For example, I didn't watch the _Avengers_ series before doing this analysis. On the other hand, the movie _Braveheart_ (1995) most probably had enough time to get high numbers.

Each of these hypotheses could warrant an investigation, and there might be other ideas that you can come up with yourself. Feel free to explore any of these hypotheses further on your own. Remember that practicing your skills by following your interests is one of the best ways to learn new skills and keep them sharp.

For this data analysis example, let's continue by investigating a slightly different question:

> What have people watched (or rated) most since 2000?

<h2 id="explore-a-question">Explore a Question</h2>

For this question, let's focus on the genres with a high volume of movies. You are going to identify the top 6 genres with the highest number of movies in them, and filter them out to produce the next chart:

```python
# top 6 genres by the total number of movies
top6_genre = (tidy_movie_ratings.iloc[:, 4:-1] # get the genre columns only
              .sum() # sum them up
              .sort_values(ascending=False) # sort descending
              .head(6) # get the first 6
              .index.values # get the genre names
              )

top6_genre
```

Unless the movie industry changed significantly in the time between writing this article and when you are reading it, your output will probably look like this:

```python
array(['Drama', 'Thriller', 'Action', 'Comedy', 'Adventure', 'Sci-Fi'],
      dtype=object)
```

Now, you want to get the ratings for these genres from your `tidy_movie_ratings` data frame, but restrict the ratings to only the movies made between 2000 and 2019:

```python
genre_groups = (tidy_movie_ratings.iloc[:, 4:]
                .groupby("production_year")
                .sum()
               ).loc["2000":"2019", top6_genre] # since 2000
```

Finally, you can create a graph showing a 2-year moving average of the total volume of rated films:

```python
genre_groups.rolling(2).mean().plot(figsize=(15,5),
                                    title="Total Rated Films")
```
&nbsp;

And here is your graph output for this data:

![2-year moving average plot for total rated films](https://github.com/CodingNomads/articles/blob/main/movie-tweets/imgs/output_69_0.png?raw=true)

This gives a nice visual representation and helps you to interpret the data to answer the question you posed before. Here are the take-aways that I took from it:

- _Drama_ and _Thriller_ are the winner genres
- Seems that _Sci-Fi_ & _Adventure_ are not as popular

On the other hand, some patterns can be misleading since we are only looking at the absolute numbers. Therefore, another way to analyze this phenomenon would be to look at the _percentage changes_. This could help your decision making if you are, let's say, in the business of online movie streaming.

So let's give that a try and plot the percentage changes:

```python
percent_change = (tidy_movie_ratings.iloc[:, 4:]
                    .groupby("production_year")
                    .sum()
                    .pct_change(periods=2) # 2 years percent change of the volume
                   ).loc["2000":"2019", top6_genre]
```

From this filtered data, let's produce a 5-years moving average graph:

```python
(percent_change.rolling(5).mean() # 5 years moving average
 .plot(figsize=(15,5),
       title="Percentage Change in Rated Films"))
```
&nbsp;

And the output is shown below:

![5-year moving average plot for percentage changes](https://github.com/CodingNomads/articles/blob/main/movie-tweets/imgs/output_72_0.png?raw=true)

You notice the decline you already spotted earlier. However, it's interesting to see the _Sci-Fi_ & _Adventure_ genres moving to the top.

Indeed, _Sci-Fi_ & _Adventure_ movies were a real _hype_, and you might want to play your cards into them, especially if your business is somewhat related to global film industry trends. These two genres has the sharpest slope for the increase in receiving ratings. This _may_ signal that there is an increasing demand and could be a valuable insight for your business.

Let's stay with one of these hyped genres for a bit longer and explore yet another question you can answer through this data set.

<div style="width:100%;height:0;padding-bottom:41%;position:relative;"><iframe src="https://giphy.com/embed/26BRzQS5HXcEWM7du" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/martial-arts-come-here-morpheus-26BRzQS5HXcEWM7du">via GIPHY</a></p></div>
&nbsp;

<h3 id="top-rated-sci-fi-movies-by-decades">Top Rated Sci-Fi Movies by Decades</h3>

Let's say you're still building out your imaginary streaming service, you understood that the interest in Sci-Fi movies is rising sharply, and you want to make it easy for your users to find the best Sci-Fi movies of all times. What are the movies _from each decade_ which you could suggest to your users by default?

To answer this question, let's start by writing the necessary steps:

- Build a `scifi` base table containing only the columns you need
- Filter for the records before 2020
- Create a new column called `decade`
- Check it out

And here's the code to accomplish these tasks:

```python
cols = ["movie_title", "rating", "production_year", "Sci-Fi", "movie_id"]
condition0 = tidy_movie_ratings["production_year"].astype(int) < 2020
condition1 = tidy_movie_ratings["Sci-Fi"] == 1

scifi = (tidy_movie_ratings
         [cols]
         [condition0 & condition1]
         .drop("Sci-Fi", axis=1)
        )

scifi["decade"] = scifi['production_year'].astype(int)//10*10

scifi.head()
```
&nbsp;

The first 5 rows of your new `scifi` data frame will look like this:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_title</th>
      <th>rating</th>
      <th>production_year</th>
      <th>movie_id</th>
      <th>decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>A Trip to the Moon</td>
      <td>7</td>
      <td>1902</td>
      <td>417</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A Trip to the Moon</td>
      <td>10</td>
      <td>1902</td>
      <td>417</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A Trip to the Moon</td>
      <td>8</td>
      <td>1902</td>
      <td>417</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A Trip to the Moon</td>
      <td>8</td>
      <td>1902</td>
      <td>417</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A Trip to the Moon</td>
      <td>10</td>
      <td>1902</td>
      <td>417</td>
      <td>1900</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

Next, you will filter for movies that have more than 10 ratings. But how can you find how many times a movie was rated? Here `.groupby()` comes to the rescue. After getting the counts, you will generate a new list called `movie_list` with the condition that a movie needs to have greater than 10 ratings. Below, the final operation will be only about getting the indices of the filtered `count_group`. You will achieve that by using `.index.values` method:

```python
count_group = scifi.groupby("movie_id").count()["rating"]

movie_list = count_group[count_group > 10].index.values
movie_list[:5]
```

The output looks like below:

```python
array([  417, 17136, 21884, 24184, 24216])
```

`movie_list` now contains those movies that have been rated more than 10 times. Next, you will filter on your `scifi` base table using the `movie_list`. Notice the usage of the `.isin()` method. It is quite user-friendly and straight-forward:

```python
condition = scifi["movie_id"].isin(movie_list)
columns = ["movie_title", "decade", "rating"]

scifi_filtered = scifi[condition][columns]
```
&nbsp;

After you created the `filtered_scifi` table, you can focus on building up your metrics in order to select the best liked movies of each decade. You will look at the average rating, and you will need to `.groupby()` decade and `movie_title`.

It is important to sort the aggregated value in a _descending_ order to get the results you are expecting. You want each group to have a maximum of 5 films, so a _lambda expression_ can help you to loop through the decade groups and show only the top 5. Otherwise, if there are less than 5 films in a decade, you want to show only the top movie, meaning only 1 record. Finally you will `round` the ratings to two decimal points.

You are encouraged to chop the code shown below into single lines and see the individual result for each of them:

```python
top_rate_by_decade = (scifi_filtered
                     .groupby(["decade", "movie_title"])
                     .mean()
                     .sort_values(["decade", "rating"],
                                                ascending=False)
                     .groupby(level=0, as_index=False)
                     .apply(lambda x: x.head() if len(x) >= 5 else x.head(1))
                     .reset_index(level=0, drop=True)
                    ).round(2)

top_rate_by_decade
```
&nbsp;

The output of this operation will be your top-rated _Sci-Fi_ movies by decade:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>decade</th>
      <th>movie_title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900</th>
      <th>A Trip to the Moon</th>
      <td>8.48</td>
    </tr>
    <tr>
      <th>1920</th>
      <th>Metropolis</th>
      <td>8.73</td>
    </tr>
    <tr>
      <th>1930</th>
      <th>King Kong</th>
      <td>8.64</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1950</th>
      <th>The Day the Earth Stood Still</th>
      <td>8.45</td>
    </tr>
    <tr>
      <th>Forbidden Planet</th>
      <td>8.40</td>
    </tr>
    <tr>
      <th>Invasion of the Body Snatchers</th>
      <td>8.16</td>
    </tr>
    <tr>
      <th>Kiss Me Deadly</th>
      <td>8.00</td>
    </tr>
    <tr>
      <th>Creature from the Black Lagoon</th>
      <td>7.91</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1960</th>
      <th>La jetée</th>
      <td>8.56</td>
    </tr>
    <tr>
      <th>Planet of the Apes</th>
      <td>8.28</td>
    </tr>
    <tr>
      <th>The Time Machine</th>
      <td>8.20</td>
    </tr>
    <tr>
      <th>2001: A Space Odyssey</th>
      <td>8.11</td>
    </tr>
    <tr>
      <th>Alphaville, une étrange aventure de Lemmy Caution</th>
      <td>7.72</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1970</th>
      <th>Alien</th>
      <td>8.47</td>
    </tr>
    <tr>
      <th>Stalker</th>
      <td>8.36</td>
    </tr>
    <tr>
      <th>Star Wars</th>
      <td>8.35</td>
    </tr>
    <tr>
      <th>Solaris</th>
      <td>8.35</td>
    </tr>
    <tr>
      <th>A Clockwork Orange</th>
      <td>8.34</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1980</th>
      <th>Back to the Future</th>
      <td>8.94</td>
    </tr>
    <tr>
      <th>The Return of the Living Dead</th>
      <td>8.71</td>
    </tr>
    <tr>
      <th>Star Wars: Episode V - The Empire Strikes Back</th>
      <td>8.66</td>
    </tr>
    <tr>
      <th>Aliens</th>
      <td>8.64</td>
    </tr>
    <tr>
      <th>E.T. the Extra-Terrestrial</th>
      <td>8.46</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1990</th>
      <th>Terminator 2: Judgment Day</th>
      <td>9.10</td>
    </tr>
    <tr>
      <th>Gekijô-ban poketto monsutâ - Myûtsû no gyakushû</th>
      <td>8.83</td>
    </tr>
    <tr>
      <th>Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni</th>
      <td>8.65</td>
    </tr>
    <tr>
      <th>The Matrix</th>
      <td>8.62</td>
    </tr>
    <tr>
      <th>The Truman Show</th>
      <td>8.53</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2000</th>
      <th>Cowboy Bebop: Tengoku no tobira</th>
      <td>9.07</td>
    </tr>
    <tr>
      <th>The Prestige</th>
      <td>8.88</td>
    </tr>
    <tr>
      <th>WALL·E</th>
      <td>8.70</td>
    </tr>
    <tr>
      <th>V for Vendetta</th>
      <td>8.44</td>
    </tr>
    <tr>
      <th>2046</th>
      <td>8.40</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2010</th>
      <th>Avengers: Endgame</th>
      <td>9.04</td>
    </tr>
    <tr>
      <th>Inception</th>
      <td>9.02</td>
    </tr>
    <tr>
      <th>Interstellar</th>
      <td>8.84</td>
    </tr>
    <tr>
      <th>Avengers: Infinity War</th>
      <td>8.76</td>
    </tr>
    <tr>
      <th>Boku no hîrô akademia THE MOVIE ~ 2-ri no eiyû ~</th>
      <td>8.62</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

If you want to see the values starting from 1990, you can do so by slicing the data frame:

```python
# loc method for filtering with the index
top_rate_by_decade.loc[1990:]
```
&nbsp;

Here are the results going back to 1990:

<div class="table-div" style="overflow-x: scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>decade</th>
      <th>movie_title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1990</th>
      <th>Terminator 2: Judgment Day</th>
      <td>9.10</td>
    </tr>
    <tr>
      <th>Gekijô-ban poketto monsutâ - Myûtsû no gyakushû</th>
      <td>8.83</td>
    </tr>
    <tr>
      <th>Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni</th>
      <td>8.65</td>
    </tr>
    <tr>
      <th>The Matrix</th>
      <td>8.62</td>
    </tr>
    <tr>
      <th>The Truman Show</th>
      <td>8.53</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2000</th>
      <th>Cowboy Bebop: Tengoku no tobira</th>
      <td>9.07</td>
    </tr>
    <tr>
      <th>The Prestige</th>
      <td>8.88</td>
    </tr>
    <tr>
      <th>WALL·E</th>
      <td>8.70</td>
    </tr>
    <tr>
      <th>V for Vendetta</th>
      <td>8.44</td>
    </tr>
    <tr>
      <th>2046</th>
      <td>8.40</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2010</th>
      <th>Avengers: Endgame</th>
      <td>9.04</td>
    </tr>
    <tr>
      <th>Inception</th>
      <td>9.02</td>
    </tr>
    <tr>
      <th>Interstellar</th>
      <td>8.84</td>
    </tr>
    <tr>
      <th>Avengers: Infinity War</th>
      <td>8.76</td>
    </tr>
    <tr>
      <th>Boku no hîrô akademia THE MOVIE ~ 2-ri no eiyû ~</th>
      <td>8.62</td>
    </tr>
  </tbody>
</table>
</div>
&nbsp;

<div style="width:100%;height:0;padding-bottom:55%;position:relative;"><iframe src="https://giphy.com/embed/ckeHl52mNtoq87veET" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/yes-avengers-infinity-war-ckeHl52mNtoq87veET">via GIPHY</a></p></div>
&nbsp;

<h2>Success! Your very own data analysis example project</h2>

**Congratulations!** You have officially completed your first movie recommendation engine! Ok, I know it's not quite Netflix - which uses machine learning to recommend what _you_ should watch. However in the tables you just generated, you've established some rule-of-thumb recommendations based on data and logic - a solid and fun first step!

What's more, you've completed your own full data analysis example project:

- You read your data as pandas data frames
- You created basic statistics and interpreted the results
- You joined data frames, applied conditions to filter them, and aggregated them
- You used data visualization to find patterns and develop hypotheses
- And you didn't jump into conclusions and root causes. You kept your reasoning simple and skeptic
- You created summary tables

All of the above are important and common aspects of working with data.

<h2 id="source-code"><a href="https://github.com/CodingNomads/movie-analysis-python-pandas" target="_blank">Source Code on GitHub</h2>

<h2 id="what-next">What Next?</h2>

If you enjoyed this data analysis example and you want to learn more and practice your skills further:

- **Add More Data**: You can search for some additional IMDB data freely available on the internet. Chances are they contain information about _directors_ of the movies. You could join this data with your `tidy_movie_ratings` dataset and see which directors are getting top ratings for which movies over the years, and by decades. This way, you can practice everything you have learned here over again
- **Build Your Service**: You can write a function which takes the `top_rate_by_decade` data frame as input and returns a random movie from the list, further simulating a movie recommendation system
- **Your Idea Here**: There are limitless possibilities to practice and play with this data. Share your explorations with us if you do!
- **If you want to learn more**: Check out CodingNomads' <a href="https://codingnomads.co/courses/data-science-machine-learning-course" target="_blank">Data Science & Machine Learning Course</a> to dive even deeper into data analysis and run full end-to-end machine learning projects on your own!

I hope you enjoyed this article and continue having fun with analyzing your datasets.

---

> **About the Author:** **Cagdas Yetkin** is a Data Scientist at Nokia where he works on the next generation Supply Chain improvements for network devices and software. He develops soccer analytics and betting applications as a hobby, and enjoys traveling. Connect with him on <a href="https://www.linkedin.com/in/cagdasyetkin/" target="_blank">LinkedIn</a> and <a href="https://twitter.com/cagdasyetkin" target="_blank">Twitter</a>.

Editor: Martin Breuss, <a href='https://twitter.com/martinbreuss' target='_blank'>@martinbreuss</a>, <a href='https://martinbreuss.com/' target='_blank'>martinbreuss.com</a>
