import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu

def get_bleu(gt1, candidate, vocab):
    gt = gt1.cpu().detach().numpy()
    score_full = 0
    score_1 = 0
    score_2 = 0
    score_3 = 0
    score_4 = 0
    for i in range(gt.shape[0]):    
        sampled_caption = []
        for word_id in candidate[i]:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence_can = ' '.join(sampled_caption)
        
        gt_caption = []
        for word_id in gt[i]:
            word = vocab.idx2word[word_id]
            gt_caption.append(word)
            if word == '<end>':
                break
        sentence_gt = ' '.join(gt_caption)
        score_full += sentence_bleu([sentence_gt], sentence_can)
        score_1 += sentence_bleu([sentence_gt], sentence_can, weights = (1,0,0,0))
        score_2 += sentence_bleu([sentence_gt], sentence_can, weights = (0,1,0,0))
        score_3 += sentence_bleu([sentence_gt], sentence_can, weights = (0,0,1,0))
        score_4 += sentence_bleu([sentence_gt], sentence_can, weights = (0,0,0,1))
    score_full /= gt.shape[0]
    score_1 /= gt.shape[0]
    score_2 /= gt.shape[0]
    score_3 /= gt.shape[0]
    score_4 /= gt.shape[0]
    
    return [score_full, score_1, score_2, score_3, score_4]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size, shuffle=True, 
                             num_workers=args.num_workers)
    
    val_loader = get_loader(args.val_image_dir, args.val_caption_path, vocab, 
                            transform, args.batch_size, shuffle=False, 
                            num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    train_loss_arr = []
    val_loss_arr = []
    train_bleu_arr = []
    val_bleu_arr = []
    for epoch in range(1, args.num_epochs+1, 1):
        iteration_loss = []
        iteration_bleu = []
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            #print(outputs.shape, targets.shape)
            loss = criterion(outputs, targets)
            iteration_loss.append(loss.item())
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            
            #get BLEU score for corresponding batch
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids.cpu().numpy()
            bleu_score_batch = get_bleu(captions, sampled_ids,vocab)
            iteration_bleu.append(bleu_score_batch)

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Bleu: '
                      .format(epoch, args.num_epochs, i, total_step, loss.item())+ str(bleu_score_batch))
                f_log = open(os.path.join(args.model_path, "log.txt"),"a+")
                f_log.write("Epoch: " + str(epoch) + "/" + str(args.num_epochs) + " Step: " + str(i) + "/" + 
                            str(total_step) + " loss: "+ str(loss.item()) + " Bleu: " + str(bleu_score_batch)+"\n") 
                f_log.close()
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                     args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1))) 
                torch.save(encoder.state_dict(), os.path.join(
                     args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1))) 
        
        train_loss_arr.append(np.array(iteration_loss))
        train_bleu_arr.append(np.array(iteration_bleu))
        
        val_loss = 0
        val_steps = 0
        val_iteration_loss = []
        val_iteration_bleu = []
        for j, (images_val, captions_val, lengths_val) in enumerate(val_loader):
            
            # Set mini-batch dataset
            images_val = images_val.to(device)
            captions_val = captions_val.to(device)
            targets = pack_padded_sequence(captions_val, lengths_val, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images_val)
            outputs = decoder(features, captions_val, lengths_val)
            #print(outputs.shape, targets.shape)
            loss = criterion(outputs, targets).item()
            val_loss += loss
            val_iteration_loss.append(loss)
            val_steps += 1
            
            #get BLEU score for corresponding batch
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids.cpu().numpy()
            bleu_score_batch = get_bleu(captions_val, sampled_ids,vocab)
            val_iteration_bleu.append(bleu_score_batch)
            
        val_loss /= val_steps
        print('Epoch [{}/{}], Val Loss: {:.4f}, Bleu: '
              .format(epoch, args.num_epochs, val_loss)+ str(bleu_score_batch))
        f_log = open(os.path.join(args.model_path, "log.txt"),"a+")
        f_log.write("Epoch: " + str(epoch) + "/" + str(args.num_epochs) + 
                    " val loss: " + str(val_loss) + " Bleu: " + str(bleu_score_batch)+"\n\n") 
        f_log.close()
        val_loss_arr.append(np.array(val_iteration_loss))
        val_bleu_arr.append(np.array(val_iteration_bleu))
        
    np.save(os.path.join(args.model_path, "train_loss.npy"), np.array(train_loss_arr))
    np.save(os.path.join(args.model_path, "val_loss.npy"), np.array(val_loss_arr))
    np.save(os.path.join(args.model_path, "train_bleu.npy"), np.array(train_bleu_arr))
    np.save(os.path.join(args.model_path, "val_bleu.npy"), np.array(val_bleu_arr))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='exp2/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--val_image_dir', type=str, default='data/val_resized2014', help='directory for resized val images')
    parser.add_argument('--caption_path', type=str, default='annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str, default='annotations/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=300, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
