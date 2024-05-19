# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:

        optimizer.zero_grad() # Zeroing the gradient
        # print(sample['utterances'])
        slots = model(sample['utterances'], sample["attentions"], sample["token_type_ids"])
        loss_slot = criterion_slots(slots, sample['y_slots'])

        loss_array.append(loss_slot.item())
        loss_slot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_array