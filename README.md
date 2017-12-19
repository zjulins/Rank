# Rank
a rank system for KBP2017's EDL task
We use simply a three layers fully connected network to get the score for each candidate,the input contains several features we extract from the mention and the candidate entity:
1.word embedding of mention and candidate entity;
2.the type for mention and candidate entity extracted from the downstream NER task and the freebase respectively;
3.the node hot of candidate entity;
4.the TF-IDF cos-similarity;
5.the surface name string similarity between the mention and the candidate entity.
