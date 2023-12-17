import numpy as np
import torch
from torch import nn

from vit import ViT


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dicts = [unpickle(f"cifar-10/data_batch_{i}") for i in range(1, 6)]
images = np.concatenate([d[b'data'].reshape(-1, 3, 32, 32) for d in dicts], axis=0)
labels = np.concatenate([np.array(d[b'labels']) for d in dicts], axis=0)

test_data = unpickle(f"./cifar-10/test_batch")
test_images = test_data[b'data'].reshape(-1, 3, 32, 32)
test_labels = np.array(test_data[b'labels'])

images = torch.tensor(images, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float)
test_labels = torch.tensor(test_labels, dtype=torch.long)


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    epochs = 200
    lr = 1e-3
    batch_size = 128
    
    data_mean = [125.3069, 122.9501, 113.8660]
    data_std = [62.9932, 62.0887, 66.7049]
    
    adamConfig = {
        "lr": 1e-3,
        "weight_decay": 5e-5,
    }

    cosineLrConfig = {
        "T_max": epochs,
        "eta_min": 1e-5,
    }

    stepLrConfig = {
        "gamma":0.95,
        "step_size":1,
    }
    
    ViTConfig = {
        "image_size": [32, 32],
        "patch_size": 4,
        "dim": 256, # try 64 later
        "head_num": 8,
        "head_dim": 64, # feat dim = head_num * head_dim = 256
        "ffn_hidden": 256,
        "layer_num": 7,
        "class_num": 10,
        "dropout": 0.1,
        "channels": 3,
        "feat_extract": "class",
    }

    model_dir = "model_pth/"


def data_transform(batch_images, batch_labels, data_mean, data_std):
    batch_images = (batch_images-data_mean) / data_std
    return batch_images, batch_labels


def get_data_iter(images, labels, batch_size, data_mean, data_std):
    data_mean = torch.tensor(data_mean).view((1, -1, 1, 1))
    data_std = torch.tensor(data_std).view((1, -1, 1, 1))
    def _data_iter():
        index = np.arange(0, len(images), batch_size)
        np.random.shuffle(index)
        for i in index:
            yield data_transform(images[i:i+batch_size], labels[i:i+batch_size], data_mean, data_std)
    return _data_iter


def train():
    data_iter = get_data_iter(images, labels, Config.batch_size, Config.data_mean, Config.data_std)
    test_iter = get_data_iter(test_images, test_labels, Config.batch_size, Config.data_mean, Config.data_std)

    model = ViT(**Config.ViTConfig).to(Config.device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **Config.cosineLrConfig)

    for epoch in range(Config.epochs):
        total_acc, total_cnt, total_loss, iter_len = 0, 0, 0, 0
        for image, label in data_iter():
            image, label = image.to(Config.device), label.to(Config.device)

            logits = model(image)
            pred_label = logits.argmax(dim=-1).to(torch.float32)
            loss = loss_fn(logits, label)

            acc = (pred_label == label).sum().item()
            total_acc = total_acc + acc
            total_cnt = total_cnt + len(label)
            total_loss = total_loss + loss
            iter_len = iter_len + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\rEpoch {epoch}| Acc {acc/len(label):.4f}| Loss {loss:.4f}| lr {lr_scheduler.get_last_lr()[0]:.6f}", end='')
        lr_scheduler.step()
        print(f"[Epoch finish] average acc {total_acc/total_cnt:.4f}| average loss {total_loss/iter_len:.4f}")
        torch.save(model.state_dict(), Config.model_dir+f"model-{epoch}.pth")

        with torch.no_grad():
            model.eval()
            total_cnt, total_acc = 0, 0
            for image, label in test_iter():
                image, label = image.to(Config.device), label.to(Config.device)

                logits = model(image)
                pred_label = logits.argmax(dim=-1).to(torch.float32)

                acc = (pred_label == label).sum().item()
                total_acc = total_acc + acc
                total_cnt = total_cnt + len(label)
            print(f"[Eval finish] average acc {total_acc/total_cnt:.4f}")


if __name__ == '__main__':
    train()