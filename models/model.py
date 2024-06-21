import torch
import torch.nn as nn
import torch.nn.functional as F

from data import textproc

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        #print(x.size())
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        return output, hidden

class DnqAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward_comb(self, query, keys):
        #print(query.size(), keys.size())
        scores = self.Va(F.tanh(self.Wa(query)  + self.Ua(keys)))

        scores = scores.reshape(scores.size(0), -1, 1, 2)
        keys = keys.reshape(keys.size(0), -1, 2, keys.size(-1)) 

        weights = F.softmax(scores, dim=-1)
        keys = torch.matmul(weights, keys)

        keys = keys.reshape(keys.size(0), -1, keys.size(-1))

        return keys, weights

    def forward(self, query, keys):
        #print(query.size(), keys.size())
        attn = []
        while(keys.size(1) > 1):
            keys, weights = self.forward_comb(query, keys)
            attn.append(weights)

        return keys, attn

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

        self.attention = DnqAttention(hidden_size)

    def forward_step(self, x, hidden, encoder_outputs):
        #print(x.size(), hidden.size(), encoder_outputs.size())
        embedded = self.dropout(self.embedding(x))
        
        query = hidden.permute(1, 0, 2)
        context, attn = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn
        

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        #print(encoder_outputs.size(), encoder_hidden.size())
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(textproc.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(textproc.max_len2()):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                #teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                #use own prediction
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden, attentions
