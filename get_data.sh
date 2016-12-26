#!/bin/bash

echo "Downloading WikiReading TensorFlow Records..."

wget https://github.com/dmorr-google/wiki-reading/blob/master/README.md
wget https://github.com/dmorr-google/wiki-reading/blob/master/data/answer.vocab
wget https://github.com/dmorr-google/wiki-reading/blob/master/data/document.vocab
wget https://github.com/dmorr-google/wiki-reading/blob/master/data/raw_answer.vocab
wget https://github.com/dmorr-google/wiki-reading/blob/master/data/type.vocab
wget -c https://storage.googleapis.com/wikireading/validation.json.tar.gz
tar xvzf validation.json.tar.gz &
wget -c https://storage.googleapis.com/wikireading/test.json.tar.gz
tar xvzf test.json.tar.gz &
wget -c https://storage.googleapis.com/wikireading/train.json.tar.gz
tar xvzf train.json.tar.gz

echo "Done."
