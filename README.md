# CES_GPT_NLP_ideology

This repository holds multiple methods of natural language processing to predict a CES respondent's self-reported three-way party ID. 

Data can be found at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SWMWX8.

Ideology is coded as a trinary, wherein 0 is liberal, 1 is moderate, 2 is conservative. Respondents who indicated a different ideology or said they were unsure were removed from the analysis to reduce noise. 


CC15_300 is an open-ended text response to the question: "When it comes to politics today, how would you describe yourself?"


### CES_BERT_model_v1.py ##
Uses BERT NLP to to construct a supervised model. Accuracy: 46%.  

### CES_NB_model_v1.py ##
Uses a Bernoulli Naïve Bayes model. Accuracy: 62%.

### CES_RNN_model_v1.py ##
Uses a recurrent neural network with TensorFlow. Accuracy: 59%. 

### CES_RNN_model_v2.py ##
Uses a recurrent neural network with TensorFlow. Uses word embeddings from https://huggingface.co/fse/fasttext-wiki-news-subwords-300/blob/main/fasttext-wiki-news-subwords-300.model.vectors.npy.
Accuracy: 61%. 
Performance shown in CES_RNN_model_v2_performance.py 



Subsequent work will be done using Davinci-003, the technology behind OpenAI's Chat-GPT to predict a respondent's ideology.  
