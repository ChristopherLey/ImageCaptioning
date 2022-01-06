import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from coco_dataset_tools import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from model import EncoderCNN, DecoderRNN

transform_test = transforms.Compose([
    transforms.Resize(257),  # smaller edge of image resized to 257
    transforms.RandomCrop(256),  # get 256x256 crop from random location
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create the data loader.
data_loader = get_loader(transform=transform_test,
                         mode='test')

for i in range(2):
    # Obtain sample image before and after pre-processing.
    orig_image, image = next(iter(data_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_file = "encoder-2021-12-17-5.pkl"
    decoder_file = "decoder-2021-12-17-5.pkl"

    embed_size = 1024  # dimensionality of image and word embeddings
    hidden_size = 1024  # number of features in hidden state of the RNN decoder

    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, device=device)
    decoder.eval()

    # Load the trained weights.
    encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
    decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)

    # Move image Pytorch Tensor to GPU if CUDA is available.
    image = image.to(device)

    with torch.no_grad():
        # Obtain the embedded image features.
        features = encoder(image)
        print(features)

        # Pass the embedded image features through the model to get a predicted caption.
        output = decoder.sample(features, debug=True)
        print('example output:', output)

    sentence = ""
    for word in output:
        sentence += data_loader.dataset.vocab.idx2word[word] + " "

    plt.figure()
    plt.imshow(np.squeeze(orig_image))
    plt.title(sentence)
plt.show()