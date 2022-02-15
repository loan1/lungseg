#importing the libraries
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
def training_loop(model, optimizer, criterion, scheduler, device, num_epochs, dataloader, CHECKPOINT_PATH, opt):
    model.to(device)
    #List to store loss to visualize
    lossli = []
    accli = []
    
    valid_loss_min = np.Inf # track change in validation loss
    count = 0
    patience = 8 # nếu val_loss tăng 8 lần thì ngừng
    for epoch in range(num_epochs):
        
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        train_acc = 0.0
        valid_acc = 0.0
        
        ###################
        # train the model #
        ###################
        
        model.train()
        for data, mask in tqdm(dataloader(opt)['train']):
            data = data.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = torch.nn.Sigmoid()(output)

            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
        
            
            train_loss += loss.item()*data.size(0)
            
            _, pred = torch.max(output, 1)              
            
            train_acc += pred.eq(mask).sum().item()
            
        scheduler.step() ###########
            
        ######################
        # validate the model #
        ######################
        
        model.eval()
        with torch.no_grad():
            for data, mask in tqdm(dataloader(opt)['val']):
                data = data.to(device)
                mask = mask.to(device)
                output = model(data)
                output = torch.nn.Sigmoid()(output)
                
                loss = criterion(output, mask)
                valid_loss += loss.item()*data.size(0)
                
                # Calculate accuracy
                _, pred = torch.max(output, 1)
#                 y_true += target.tolist()
#                 y_pred += pred.tolist()  
                
                valid_acc +=  pred.eq(mask).sum().item()
       
        # calculate average losses
        train_loss = train_loss/len(dataloader()['train'].dataset)
        valid_loss = valid_loss/len(dataloader()['val'].dataset)
        lossli.append({'epoch':epoch,'train_loss': train_loss,'valid_loss':valid_loss})
        
        train_acc = train_acc*100/len(dataloader()['train'].dataset)
        valid_acc = valid_acc*100/len(dataloader()['val'].dataset)
        accli.append({'epoch':epoch,'train_acc': train_acc,'valid_acc':valid_acc})
        
        ####################
        # Early stopping #
        ##################
        
        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \n \tTraining Acc: {:.6f} \tValidation Acc: {:.6f}'.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc))
        # save model if validation loss has decreased
       
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': accli,
            'loss_list': lossli,
            'loss': loss
            }, CHECKPOINT_PATH)
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            
            count = 0
            print('count = ',count)
            torch.save(model, './model/FTResNet50.pt') #save model 
                                  
            valid_loss_min = valid_loss
        else:
            count += 1
            print('count = ',count)
            if count >= patience:
                print('Early stopping!')
   
                return lossli, accli    
           
    return lossli, accli