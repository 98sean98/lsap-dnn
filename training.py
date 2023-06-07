import os
import mlflow
import numpy as np
import h5py
import torch


# only if running an experiment outside deploifai (this must come before setting mlflow tracking uri and experiment)
os.environ["MLFLOW_TRACKING_USERNAME"] = "Deploifai/LSAP/linear_model"
os.environ["MLFLOW_TRACKING_PASSWORD"] = ""

# setup mlflow for this experiment
mlflow.set_tracking_uri('https://community.mlflow.deploif.ai')
mlflow.set_experiment("Deploifai/LSAP/linear_model")

torch.manual_seed(583)

device = 'cuda'

def main(log_metric, log_param):
    n = 4
    batch_size = 512
    learning_rate = 1e-3
    l2_regularization_weight = 1e-5
    epochs = 100
    
    log_param('n', 4)
    log_param('batch_size', batch_size)
    log_param('learning_rate', learning_rate)
    log_param('l2_regularization_weight', l2_regularization_weight)
    log_param('epochs', epochs)
    
    
    data_file = h5py.File('data/data_4_100_50000.h5', 'r')
    x_data = data_file['x']
    y_data = data_file['y']
    x_bound = 100
    
    x_data = np.array(x_data, dtype='float32')
    y_data = np.array(y_data, dtype=int)
    
    ys = []
    for y in y_data:
        a = np.zeros((n, n), dtype='float32')
        a[y[:,0], y[:,1]] = 1
        ys.append(a)
    ys = torch.from_numpy(np.array(ys))
    xs = torch.from_numpy(x_data)
    
    x_train, x_test = xs[:45000], xs[45000:]
    y_train, y_test = ys[:45000], ys[45000:]
    
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    
    models = [LinearModel(n * n, n).to(device) for _ in range(n)]
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization_weight)
        for model in models
    ]
    
    for e in range(epochs):
        report_loss = e % 10 == 0
        
        training_losses = train_one_epoch(train_dataloader, train_size, models, loss_fn, optimizers)
        test_losses, accuracy = test_one_epoch(test_dataloader, test_size, models, loss_fn)
        
        if report_loss:
            print("epoch:", e)
            print("training_losses:", training_losses)
            print("test_losses:", test_losses)
            print("accuracy:", accuracy)
            
        for f, l in enumerate(training_losses):
            log_metric(f"training_loss_{f}", l)
        for f, l in enumerate(test_losses):
            log_metric(f"test_loss_{f}", l)
        for f, a in enumerate(accuracy):
            log_metric(f"accuracy_{f}", a)

    
    models_dict = {}
    for i, model in enumerate(models):
        models_dict[f'model_{i}'] = model.state_dict()
    
    torch.save(models_dict, 'artifacts/models.pt')
    
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs.to(device)
        self.ys = ys.to(device)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class LinearModel(torch.nn.Module):
    def __init__(self, input_n, output_n):
        super().__init__()
        
        self.stack = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_n, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, output_n),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.stack(x)

    
def transform_ys(ys):
    ys = torch.swapaxes(ys, 0, 1)
    ys = torch.split(ys, 1)
    return [k[0] for k in ys]


def train_one_epoch(train_dataloader, train_size, models, loss_fn, optimizers):
    train_losses = [0 for _ in models]
    
    for i, data in enumerate(train_dataloader):
        xs, ys = data
        batch_size = xs.shape[0]

        for optimizer in optimizers:
            optimizer.zero_grad()

        ys_pred = [model(xs) for model in models]
        ys = transform_ys(ys)
        
        outputs = zip(ys_pred, ys)

        losses = [loss_fn(p, q) for p, q in outputs]

        for loss in losses:
            loss.backward()

        for optimizer in optimizers:
            optimizer.step()
            
        train_losses = zip(train_losses, losses)
        train_losses = [l1 + l2.cpu().item() * batch_size for l1, l2 in train_losses]
    
    train_losses = np.array(train_losses) / train_size
    
    return train_losses.tolist()

        
def test_one_epoch(test_dataloader, test_size, models, loss_fn):
    test_losses = [0 for _ in models]
    correct = [0 for _ in models]
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            xs, ys = data
            batch_size = xs.shape[0]
            
            ys_pred = [model(xs) for model in models]
            ys = transform_ys(ys)
            outputs = zip(ys_pred, ys)
            
            losses = [loss_fn(p, q).detach().cpu().item() for p, q in outputs]
            
            test_losses = zip(test_losses, losses)
            test_losses = [l1 + l2 * batch_size for l1, l2 in test_losses]
            
            prediction = [torch.argmax(u, dim=1) for u in ys_pred]
            label = [torch.argmax(u, dim=1) for u in ys]
            correct = zip(correct, prediction, label)
            correct = [c + (p == l).sum().item() for c, p, l in correct]
            
        test_losses = np.array(test_losses) / test_size
        accuracy = np.array(correct) / test_size

    return test_losses.tolist(), accuracy.tolist()

    
if __name__ == "__main__":
    with mlflow.start_run() as run:
        print('mlflow run id:', run.info.run_id)
        
        main(lambda k, v: mlflow.log_metric(k, v), lambda k, v: mlflow.log_param(k, v))
