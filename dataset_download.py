import urllib.request
import os
urllib.request.urlretrieve('http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/ubuntu_dialogs.tgz', "ubuntu_dialogs.tqz")
urllib.request.urlretrieve('http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip', "cornell_movie_dialogs_corpus.zip")
os.system("git clone https://github.com/google-research-datasets/Taskmaster.git")
os.system("git clone https://github.com/Meinwerk/WordPrediction.git")
urllib.request.urlretrieve('https://www.statmt.org/europarl/v7/bg-en.tgz','bg-en.tgz')