# Campaign_Classifier
This project uses a supervised machine learning model to classify Federal Election Commission data on campaign spending into one of nine categories (media, digital, polling, legal, field, consulting, fundraising, and administrative).

A major challenge working with FEC data is that campaigns and committees use a variety of terms to describe expenditures that fall within the same category of spending. Our model addresses this issue in two ways. First, we select an initial set of keywords based on our definition of each category and add additional terms that appear frequently in our training and testing data. Second, we use the Datamuse API, a word-finding query engine, to identify synonyms for our keywords in each category.

We train and test our model using the scikit-learn SGDClassifier. The classifier relies on the following packages: NLTK, NumPy, Pandas; and Python 3+. 

For additional information on the model and its application , see Sheingate, Adam; Scharf, James; Delahanty, Conner (2022): Digital Advertising in U.S. Federal Elections, 2004-2020. Preprint. https://doi.org/10.6084/m9.figshare.19372421.v2 
