import torch
import torch.nn as nn
import time

from data import datastore
from models import model
import utils

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        loss = loss_fn(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100, save_path='models/weights/tmp.pt'):
    encoder.train()
    decoder.train()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    try:
        checkpoint = torch.load(save_path)
        start_epoch = checkpoint['epoch']
        enc_model.load_state_dict(checkpoint['enc_model_state'])
        dec_model.load_state_dict(checkpoint['dec_model_state']) 
        encoder_optimizer.load_state_dict(checkpoint['enc_opt_state'])
        decoder_optimizer.load_state_dict(checkpoint['dec_opt_state'])
        print('loaded model')
    except:
        start_epoch = 1
        print('new model')

    for epoch in range(start_epoch, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.time_since(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        torch.save({
            'epoch': epoch,
            'enc_model_state': encoder.state_dict(),
            'dec_model_state': decoder.state_dict(),
            'enc_opt_state': encoder_optimizer.state_dict(),
            'dec_opt_state': decoder_optimizer.state_dict()
        }, save_path)

    utils.show_plot(plot_losses)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = 32
    batch_size = 128

    input_lang, output_lang, train_dataloader = datastore.get_dataloader('spa', 'eng', batch_size)

    enc_model = model.EncoderRNN(input_lang.n_tokens, hidden_size).to(device)
    dec_model = model.AttnDecoderRNN(hidden_size, output_lang.n_tokens).to(device)

    train(train_dataloader, enc_model, dec_model, 300, print_every=1, plot_every=1, save_path='models/weights/rnn-word-32.pt')
