import numpy as np
import torch
import torch.optim as optim

class LinearAligner():
    def __init__(self) -> None:        
        self.W = None
        self.b = None
           
    def train(self, ftrs1, ftrs2, epochs=6, target_variance=4.5, verbose=0) -> dict:
        lr_solver = LinearRegressionSolver()
        
        print(f'Training linear aligner ...')
        print(f'Linear alignment: ({ftrs1.shape}) --> ({ftrs2.shape}).')
        
        var1 = lr_solver.get_variance(ftrs1)
        var2 = lr_solver.get_variance(ftrs2)

        c1 = (target_variance / var1) ** 0.5
        c2 = (target_variance / var2) ** 0.5
        
        ftrs1 = c1 * ftrs1
        ftrs2 = c2 * ftrs2

        lr_solver.train(ftrs1, ftrs2, bias=True, epochs=epochs, batch_size=100,)
        mse_train, r2_train = lr_solver.test(ftrs1, ftrs2)
        
        print(f'Final MSE, R^2 = {mse_train:.3f}, {r2_train:.3f}')
        
        W, b = lr_solver.extract_parameters()
        W = W * c1/c2
        b = b * c1/c2
        
        self.W = W
        self.b = b   
        
    def get_aligned_representation(self, ftrs):
        return ftrs @ self.W.T + self.b
    
    def load_W(self, path_to_load: str):
        aligner_dict = torch.load(path_to_load)
        self.W, self.b = [aligner_dict[x].float() for x in ['W', 'b']]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W = self.W.to(device).float()
        self.b = self.b.to(device).float()
        
    def save_W(self, path_to_save: str):
        torch.save({'b': self.b.detach().cpu(), 'W': self.W.detach().cpu()}, path_to_save)
        
        
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out


class LinearRegressionSolver():
    def __init__(self):
        self.model = None
        self.criterion = torch.nn.MSELoss()
    
    def train(self, X: np.ndarray, y: np.ndarray, bias=True, batch_size=100, epochs=20):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tensor_X = torch.from_numpy(X).float()
        tensor_y = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.model = LinearRegression(X.shape[1], y.shape[1], bias=bias)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        self.model.to(device)

        
        init_mse, init_r2 = self.test(X, y)
        print(f'Initial MSE, R^2: {init_mse:.3f}, {init_r2:.3f}')
        
        self.init_result = init_r2
        self.model.train()

        for epoch in range(epochs):
            e_loss, num_of_batches = 0, 0

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                num_of_batches += 1
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                e_loss += loss.item()

                loss.backward()
                optimizer.step()

            e_loss /= num_of_batches
            
            print(f'Epoch number, loss: {epoch}, {e_loss:.3f}')
            
            scheduler.step()
        
        return 

     
    def extract_parameters(self):
        for name, param in self.model.named_parameters():
            if name == 'linear.weight':
                W = param.detach()
            else:
                b = param.detach()

        return W, b

    
    def get_variance(self, y: np.ndarray):
        ey = np.mean(y)
        ey2 = np.mean(np.square(y))
        return ey2 - ey**2

    
    def test(self, X: np.ndarray, y: np.ndarray, batch_size=100):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tensor_X = torch.from_numpy(X).float()
        tensor_y = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.model.eval()
        
        total_mse_err, num_of_batches = 0, 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                num_of_batches += 1
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_mse_err += loss.item()
            
        total_mse_err /= num_of_batches

        return total_mse_err, 1 - total_mse_err / self.get_variance(y)