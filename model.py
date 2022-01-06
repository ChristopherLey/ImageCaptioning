import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, encoded_image_size=14, dropout_rate=0.5):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.dropout = dropout_rate
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.encoding = nn.Linear(2048, embed_size)
        self.init_weights()

    def init_weights(self):
        self.encoding.weight.data.uniform_(-0.1, 0.1)
        self.encoding.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        pooled_features = self.adaptive_pool(features)
        flattened_feature = pooled_features.view(pooled_features.size(0), -1)
        encoded_features = self.encoding(self.dropout(flattened_feature))
        return encoded_features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate=0.5, layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout_rate

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=layers, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, encoded_captions):
        caption_lengths = encoded_captions.size(-1)
        embeddings = self.embedding(encoded_captions)[:, 0:caption_lengths - 1, :]
        encoder_out = encoder_out[:, None, :]
        decoder_input = torch.concat((encoder_out, embeddings), dim=1)
        decoder_output, _ = self.decoder(decoder_input)
        predictions = self.fc(self.dropout(decoder_output))
        return predictions

    def sample(self, encoder_out, states=None, max_len=20, debug=False):
        output = []
        greedy_input = encoder_out[:, None, :]
        decoder_output, (hidden_state, cell_state) = self.decoder(greedy_input)
        prediction_t = self.fc(decoder_output)
        predicted_word = torch.argmax(prediction_t, dim=2)
        greedy_input = self.embedding(predicted_word)
        output.append(predicted_word.item())
        for t in range(1, max_len):
            decoder_output, (hidden_state, cell_state) = self.decoder(greedy_input, (hidden_state, cell_state))
            prediction_t = self.fc(decoder_output)
            predicted_word = torch.argmax(prediction_t, dim=2)
            greedy_input = self.embedding(predicted_word)
            output.append(predicted_word.item())
        return output


if __name__ == "__main__":
    pass
