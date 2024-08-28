
import requests, re, logging
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import urllib.request, json
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import string
from string import digits
from sklearn.feature_selection import SelectFromModel

wl = WordNetLemmatizer()
sr = stopwords.words('english')

dit_url = "https://jira-oss.seli.wh.rnd.internal.ericsson.com"
dit_rest_api = "/rest/api/2/search?jql="

# stack_request = "project=CIS%20AND%20component=Openstack_Support%20AND%20created>-90d&fields=description,summary&maxResults=1000"
# cloud_request = "project=CIS%20AND%20component=\"ENM%20Test%20Environment%20Cloud\"%20AND%20created>-1d&fields=description,summary&maxResults=1000"
# netsim_request = "project=CIS%20AND%20component=Netsim-vFarm%20AND%20created>-1d&fields=description,summary&maxResults=1000"

"""
The strings used to generate the url's used to make the REST call to jira are above

dit_url + dit_rest_api + #stack_request
https://jira-oss.seli.wh.rnd.internal.ericsson.com/rest/api/2/search?jql="project=CIS%20AND%20component=Openstack_Support%20AND%20created>-90d&fields=description,summary&maxResults=1000"

Get all tickers on the CIS board with the 'Openstack_Support' label created in the past 90 days.

the fields= part of the url tells the jira API what data to send back with the result. For every issue the above query picks up, the api will return the normal data and also the full description and summary for each ticket. 
Once we sort out how you can access Jira/confluence etc I'll give one/both of you  a run through of how to interact with the JIRA api

https://developer.atlassian.com/cloud/jira/platform/rest/v3/
official docs here, if you want to take a quick look. Probably wont be very useful without being able to play around with it though.
"""

# These need to be inserted somehow
username = [jira username]
password =  [jira password]

session = requests.Session()
session.auth = (username, password)

def get_json(url):
    """
    This function gets data in json format on a given jira end point.
    """
  
    # jira returns a response object. we call response.json() to get the data contained within the respone object formatted as json
    response = session.get(url)
    jira_data = response.json()
    return jira_data

def build_jira_end_point(support, project, component, numDays, maxResults):
    """
    This function build jira and return end poin for fetching data.
    """
    jql_request = "project=" + project + "%20AND%20component=" + component + "%20AND%20created>-" + numDays + "d&fields=description,summary&startAt=0&maxResults=" + maxResults
    url = dit_url + dit_rest_api + jql_request
    jira_data = [support, get_json(url)]

    return jira_data

def extract_jira_words(jira_data):
    """
    This function extract page information
    """
    
    
    # we are taking every block of text in the summary of every ticket and appending them to one large body of text, per class/label. 

    corpus = ""
    for issue in jira_data[1]["issues"]:
        corpus += issue["fields"]["summary"] + " "
    return [jira_data[0], corpus]

def update_stop_words(stopwords):
    """
    This function updates stop words list.
    """
    
    # stopwords are like 'are, the, as, want' . conversational words which don't offer much value to a machine learning model. 
    # an idea might be to examine the text and build your own library of Ericsson/DE specific stopwords which you can use to filter out the noise from the data 
    # We did not add domain specific words / S&J
    stopwords = stopwords.words('english')
    stopwords.extend(["hello", "buy"])
    return stopwords

def assign_labels(clean_words_list, sentence):
    """
    This function assign labes to existed DataFrame
    """
    clean_words_list['component'] = sentence[0]
    return clean_words_list

def tokenize_words(corpus):
    """
    This function tokenize words and returns list of cleaned words.
    """
    words = re.sub('[^a-zA-Z \n]', ' ', corpus[1].lower())
    table = str.maketrans('','',string.punctuation)
    clean_words_list = [i.translate(table) for i in words.split()]
    remove_digits = str.maketrans('','', digits)
    clean_words_list = [i.translate(remove_digits) for i in clean_words_list]
    clean_words_list = [str(wl.lemmatize(w)) for w in clean_words_list]

    clean_tokens = clean_words_list[:]
    for token in clean_words_list:
        if token in stopwords.words('english') or len(token) <= 2 or token == "color":
            clean_tokens.remove(token)
    list_of_ngrams = list(ngrams(clean_tokens, 3))  # http://www.albertauyeung.com/post/generating-ngrams-python/
    new_data = [' '.join(w) for w in list_of_ngrams]
    clean_tokens = pd.DataFrame(new_data)

    clean_words_df = assign_labels(clean_tokens, corpus)
 
    return clean_words_df

def collect_data(list_of_components):
    """
    This function collect data and builds DataSet.
    """
    num_days = str(365)
    max_result = str(1000)
    tokenized_words_df = pd.DataFrame()

    for component in list_of_components:
        label = str(component[0])
        issue = str(component[1])
        project = str(component[2])
        jira_data = build_jira_end_point(label, issue, project, num_days, max_result)
        corpus = extract_jira_words(jira_data)
        tokenized_words_df = tokenized_words_df.append(tokenize_words(corpus), ignore_index=True)
    tokenized_words_df.rename(columns={0: 'word'}, inplace=True)
    return tokenized_words_df

def get_labels(tokenized_words_df):
    """
    This function creates set of labels.
    """
    labels_df = pd.DataFrame(tokenized_words_df.iloc[:, 1])
    labels_df = labels_df.groupby(['component'], as_index=False).sum()
    return pd.DataFrame(labels_df)

labelencoder_y = LabelEncoder()

def get_tested_dataset(tokenized_words_df):
    """
    This function gets tested dataset for testing model.
    """
    Y_data = labelencoder_y.fit_transform(tokenized_words_df.iloc[:, 1])
    return Y_data

def encode_components(tokenized_words_df):
    """
    This function updates dataset component column into encoded data.
    """
    tokenized_words_df.iloc[:, 1] = labelencoder_y.fit_transform(tokenized_words_df.iloc[:, 1])

def get_best_classifier(vectorized_words, Y_data):
    """
    This function search for best classifiers to be applied on data.
    """
    result_cols = ["Classifier", "Accuracy"]
    result_frame = pd.DataFrame(columns=result_cols)
    models = [
        (MultinomialNB(), {}),
        (LogisticRegression(max_iter=10000), {}),
        (MLPClassifier(solver='lbfgs', activation = 'relu',alpha=1e-5,hidden_layer_sizes=(8), max_iter=100000, random_state=1, tol=0.0001 ,early_stopping=True, validation_fraction=0.1, n_iter_no_change=5),{}),
     ]
    Xtrain, Xtest, ytrain, ytest = split_training_test_sets(vectorized_words, Y_data)   

    for clf, grid_val in models:
        name = clf.__class__.__name__
        clf_model = clf.fit(Xtrain,
                            ytrain) 
        ypred = clf_model.predict(Xtest)
        accuracy = accuracy_score(ytest,ypred)
        precision = precision_score(ytest,ypred, average=None, labels=[0,1,2,3,4])#,5,6,7,8,9,10,11])
        recall = recall_score(ytest,ypred, average=None, labels=[0,1,2,3,4])#,5,6,7,8,9,10,11])
        fscore = f1_score(ytest,ypred, average=None, labels=[0,1,2,3,4])#,5,6,7,8,9,10,11])
        acc_field = pd.DataFrame([[name, accuracy * 100]], columns=result_cols)
        result_frame = result_frame.append(acc_field)
        print(str(name))
        print('Confusion matrix')
        print(confusion_matrix(ytest,ypred))
        print('Accuracy: '+str(accuracy))
        print('Precision: '+str(precision))
        print('Recall: '+str(recall))
        print('F-score: ' + str(fscore))
        print("----------")
    top_three_clf = result_frame.sort_values('Accuracy', ascending=False)['Classifier'].head(3)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")

    # we dont actually use sns or plt
    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    #plt.show()
    return result_frame, top_three_clf

def fit_transform_data(tfidfVectorizer, tokenized_words):
    """
    This function transforms data into vectors and fits into TFIDFVectoriser vectors.
    """
    return tfidfVectorizer.fit_transform(tokenized_words['word']) # .toarray() # - tfidfVectorizer gets MemoryError when toarray() is applied and the models are slower

def get_vectorizer():
    """
    This function creates TFIDFVectoriser object, transform data into numerical values.
    """
    return TfidfVectorizer()


def split_training_test_sets(vectorized_words, Y_data):
    """
    This function splits data into training and test sets. 80:20.
    """
    X_train, X_test, y_train, y_test = train_test_split(vectorized_words, Y_data, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load all data in forms of list into dataframe with according labels.
    # Create new Dataframe with ngrams_range and pre_processed words with according labels
    # populate list with new component


    """
    
    when we are getting the data from JIRA, the query we use has fields for project type (index 1) and ticket label, 
    (index 2). All DE teams jira boards are in one Project, (the CIS project) and the different boards are differentiated 
    from each other based on the labels applied to the tickets.  the stack centric board gets the tickets with the
     Openstack_Support label, netsim get Netsim-vFarm, etc.
       
    """

    list_of_components = [
        ["netsim_jira_summary_data", "CIS", "Netsim-vFarm"],
        ["stack_jira_summary_data", "CIS", "Openstack_Support"],
        ["cloud_jira_summary_data", "CIS", "\"ENM%20Test%20Environment%20Cloud\""],
        ["cloud_native_summary_data","CIS","\"DE-CloudNative\""],
        ["stratus_summary_data", "CIS", "\"CI%20Infra%20/%20CI%20Fwk\""],
     #   ["axis_summary_data","CIS","\"autodeploy/DMT\""],
     #   ["TAF_summary_data","CIS","TAF"],
     #   ["NSS_summary_data","NSS","NSS"], 
     #   ["infra_summary_data","CIS","\"Infra%20OS\""],
     #   ["S2_summary_data","CIS","\"ENM%20Test%20Environment%20Physical%20Support\""],
     #   ["S4_summary_data","CIP","TEaaS"],
     #   ["S5_summary_data","CIP","\"TEaaS-S5\""]
        ]

    # tried to add S3 via ["S3_summary_data","PNTC", "\"RNA%20Support%20Request\""], but it doesn't generate anything
    # Gitmo/Axis Ops was also hard to find (it had many different components etc and no template), therefore it is not included


    """
    This function gets the text from JIRA for each of the sublists in 'list_of_components', extracts the words from the text
    and then tokenises them. We've tried tokenising the words on their own, or using N-grams (2/3/4 grams). 

    """
    tokenized_words_df = collect_data(list_of_components)

    """
    this gets the labels from the tokenised dataset
    https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
    """
    labels_df = get_labels(tokenized_words_df)
    Y_data = get_tested_dataset(tokenized_words_df)
    encode_components(tokenized_words_df)
    vectorizer = get_vectorizer()
    vectorized_words = fit_transform_data(vectorizer, tokenized_words_df)
    # k = int(vectorized_words.shape[1]*0.75) # to choose how much to keep when using feature selection
   
   # vectorized_words = SelectKBest(chi2, k=k).fit_transform(vectorized_words, Y_data) # for feature selection
    result_frame, top_three_clf = get_best_classifier(vectorized_words, Y_data)
    print(top_three_clf)

