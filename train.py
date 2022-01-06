import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from coco_dataset_tools import get_loader
from model import EncoderCNN, DecoderRNN
import math

batch_size = 32          # batch size
vocab_threshold = 2        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 1024           # dimensionality of image and word embeddings
hidden_size = 1024          # number of features in hidden state of the RNN decoder
print_every = 100          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

transform_train = transforms.Compose([
    transforms.Resize(257),  # smaller edge of image resized to 257
    transforms.RandomCrop(256),  # get 256x256 crop from random location
    transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, device=device)

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)

# Define the loss function.
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss

parameters = list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(filter(lambda p: p.requires_grad, decoder.parameters()))

# TODO #4: Define the optimizer.
optimizer = torch.optim.Adam(params=parameters)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

import torch.utils.data as data
import numpy as np
import os
import requests
import time
from datetime import datetime

save_every = 1  # determines frequency of saving model weights
num_epochs = 5  # number of training epochs
test_every = 100
# Open the training log file.
f = open(log_file, 'w')

load_saved = False
encoder_file = "encoder-2021-12-15-5.pkl"
decoder_file = "decoder-2021-12-15-5.pkl"

if load_saved:
    # Load pre-trained weights before resuming training.
    encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
    decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

start_time = time.time()
today = datetime.today()

for epoch in range(1, num_epochs + 1):

    for i_step in range(1, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)

        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()

        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Get training statistics.
        elapsed = time.time() - start_time
        completed = (i_step * epoch) / (num_epochs * total_step)
        total_time_estimate = elapsed / completed
        remaining = total_time_estimate - elapsed

        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, Elapsed: %.2f min, Remaining: %.2f min' % (
        epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()), elapsed / 60.0, remaining / 60.0)

        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()

        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()

        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)
            if i_step == print_every:
                print("Check Encoder Grads")
                for name, param in encoder.named_parameters():
                    print(name, param.grad)
                print("Check Deccoder Grads")
                for name, param in decoder.named_parameters():
                    print(name, param.grad)
            with torch.no_grad():
                # sample
                features = encoder(images[0:1, :, :, :])
                output = decoder.sample(features)
                predicted_sentence = ""
                actual_caption = ""
                for p_word, a_word in zip(output, captions[0]):
                    predicted_sentence += data_loader.dataset.vocab.idx2word[p_word] + " "
                    actual_caption += data_loader.dataset.vocab.idx2word[a_word.item()] + " "
            print(f'predicted: "{predicted_sentence}" vs actual: "{actual_caption}"')

        # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', f'decoder-{today.date()}-{epoch:d}.pkl'))
        torch.save(encoder.state_dict(), os.path.join('./models', f'encoder-{today.date()}-{epoch:d}.pkl'))

# Close the training log file.
f.close()