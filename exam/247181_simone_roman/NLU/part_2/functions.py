# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:

        optimizer.zero_grad() # Zeroing the gradient
        # print(sample['utterances'])
        slots, intent = model(sample['utterances'])
        # loss_intent = criterion_intents(intent, sample['intents'])
        # loss_slot = criterion_slots(slots, sample['y_slots'])
        # loss = loss_intent + loss_slot

        # loss_array.append(loss.item())
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # optimizer.step()

    return loss_array