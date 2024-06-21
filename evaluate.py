import torch
import random
import numpy as np

from data import textproc, datastore
from models import model

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = datastore.tensor_from_sentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attentions = decoder(encoder_outputs, encoder_hidden)

    return datastore.sentence_from_tensor(output_lang, decoder_outputs), attentions

def evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs, n=10, show_attentions=False):
    encoder.eval()
    decoder.eval()
    for i in range(n):
        try:
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_sentence, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
            print('<', output_sentence)
            if show_attentions:
                for idx, attn_weights in enumerate(attentions):
                    print(f'{idx}---')
                    for attn in attn_weights:
                        print(attn.flatten().detach().numpy())
                    if idx == len(textproc.split_tokens(output_sentence)):
                        break
            print('')
        except: 
            pass


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=2, suppress=True)

    hidden_size = 32

    input_lang, output_lang, pairs = textproc.prepare_data('spa', 'eng')

    enc_model = model.EncoderRNN(input_lang.n_tokens, hidden_size).to(device)
    dec_model = model.AttnDecoderRNN(hidden_size, output_lang.n_tokens).to(device)

    checkpoint = torch.load('models/weights/rnn-word-32.pt')
    enc_model.load_state_dict(checkpoint['enc_model_state'])
    dec_model.load_state_dict(checkpoint['dec_model_state'])

    evaluate_randomly(enc_model, dec_model, input_lang, output_lang, pairs)
    
    print('Attentions:')
    evaluate_randomly(enc_model, dec_model, input_lang, output_lang, pairs, n=2, show_attentions=True)


    sentences = [
        'my name is tom', 
        'are you ok ?', 
        'tom is smart', 
        'you are dumb', 
        'we are clowns',
        'who is the genius ?',
        'is boston a pretty place ?',
        'are the rich evil ?',
        'i am tom i am smart who are you',
        'what is love',
        'is rain good weather ?',
        'tom does not think much',
        'when can we play ?',
        'is fun wasted time ?'
    ]
    print('Custom:')
    for sentence in sentences:
        print('>', sentence)
        try:
            output_sentence, _ = evaluate(enc_model, dec_model, sentence, input_lang, output_lang)
            print('<', output_sentence)
        except:
            print('<', '[error]')
        print('')
