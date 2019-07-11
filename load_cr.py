from toolz import pipe as p

from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd


def loadReports():
	reports = loadRawReports()
	reports['txt'] = [cleanWord2VecTags(_) for _ in reports['txt']]
	return reports

def loadRawReports():
	return pd.read_csv('cr_clean.csv', encoding = 'latin1')
	
def cleanWord2VecTags(txt):
    return txt.replace(' </s>', '.')