# MiniRNN

Next character prediction generator based on Andrej Karpathys [Mini char RNN](https://gist.github.com/karpathy/d4dee566867f8291f086).
This implementation follows Andrej's numpy version but is completely implemeted in Pytorch.
Implementation contains from-scratch RNN implementation using Pytorch tensors with goals of adding more advanced models already available in `torch.nn`

## Update:
Pytorch `LSTM` architecture now available for parallel training. Increase the complexity of the model by adding more layers and increasing hidden layers size.


# Differences

Biggest difference from Andrej's implementation is parallelism. 
Since RNN are slow in training because of their sequential nature, parallelism is implemented in the following way:
- Text is divided in chunks, each containing `sequence_length` characters
- RNN is trained in parallel on each chunk separatelly
- This does make the training process multiple times faster but also considers these chunks to be completely independent, which can slightly hurt training performance.

# Usage 

Run `train.py` from command line or just `import train.run` and run it as a function. Play around with arguments to achieve best speed/performance results.
