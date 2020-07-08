# Cruz Control

## GPT2 Models 

We currently have the following training scripts for the models:
* GPT2 Baseline Text + Fact
* Knowledge Dependent Policy Driven Neural Response Generator using Mezza Tags

### Contact

For any clarification related to the above code, please reach out to Rishi Rajasekaran (rrajasek@ucsc.edu)

## DSTC9 Baseline Code (untested)

### Response Generation 

Scripts to train Seq2Seq and Transformer models on the Amazon Topical-Chat Corpus. This code serves as the baseline for [DSTC9 Track 3](http://dialog.speech.cs.cmu.edu:8003/).

**To train**: `python3 train.py --use_knowledge --transformer --save_path transformer/`

**To test**: `python3 test.py --use_knowledge --transformer --save_path transformer/`

**To serve interactive model with TF-IDF based fact selection**: `python3 dynamic.py --use_knowledge --transformer --save_path transformer/`

### Data

The pre-processed data can be found in `data.zip`. If you would like to use a different pre-processing strategy, please download the original data from [here](https://github.com/alexa/alexa-prize-topical-chat-dataset/).

### Contact

If you experience any issues with this code, please contact me at mehrishikib@gmail.com
