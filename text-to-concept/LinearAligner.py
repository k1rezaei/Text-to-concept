import numpy as np
import torch
import torch.optim as optim

class LinearAligner():
    def __init__(self) -> None:        
        self.W = None
        self.b = None
           
    def train(self, ftrs1, ftrs2, epochs=6, target_variance=4.5, verbose=0) -> dict:
        lr_solver = LinearRegressionSolver()
        
        if verbose == 1:
            print(f'aligning from first representation space with shape ({ftrs1.shape}) to second representation space with shape ({ftrs2.shape}).')
            print('initial variance of elements in those two spaces:')
            print(f'first representation space variance: {lr_solver.get_variance(ftrs1)}')
            print(f'second representation space variance: {lr_solver.get_variance(ftrs2)}')

        var1 = lr_solver.get_variance(ftrs1)
        var2 = lr_solver.get_variance(ftrs2)

        c1 = (target_variance / var1) ** 0.5
        c2 = (target_variance / var2) ** 0.5

        self.ftrs1 = c1 * ftrs1
        self.ftrs2 = c2 * ftrs2

        lr_solver.train(ftrs1, ftrs2, bias=True, epochs=epochs, batch_size=100, verbose=verbose)
        mse_train, r2_train = lr_solver.test(ftrs1, ftrs2)
        
        if verbose == 1:
            print(f'final (mse, r2): {mse_train, r2_train}')
        
        mse_ftrs, r2_ftrs = lr_solver.test_each_feature(ftrs1, ftrs2)

        var1 = lr_solver.get_variance_each_feature(ftrs1)
        var2 = lr_solver.get_variance_each_feature(ftrs2)

        W, b = lr_solver.extract_parameters()
        W = W * c1/c2
        b = b * c1/c2

        self.W = W
        self.b = b   
        return {
            'mse_train': mse_train,
            'R2_train': r2_train,
            'mse_train_each_features': mse_ftrs,
            'r2_train_each_features': r2_ftrs,
            'model1_var_each_feature': var1,
            'model2_var_each_feature': var2,
            'W': W,
            'b': b,
        }
        
        
    def get_aligned_representation(self, ftrs):
        return ftrs @ torch.transpose(self.W, 0, 1) + self.b
    
        
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
    
    def train(self, X: np.ndarray, y: np.ndarray, bias=True, batch_size=100, epochs=20, verbose=0):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tensor_X = torch.from_numpy(X).float()
        tensor_y = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.model = LinearRegression(X.shape[1], y.shape[1], bias=bias)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        self.model.to(device)

        if verbose == 1:
            init_mse, init_r2 = self.test(X, y)
            print(f'initial (mse, r2): {init_mse, init_r2}')
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
            if verbose == 1:
                print(f'(epoch, loss): {epoch, e_loss}')
            
            scheduler.step()
        
        return 

     
    def extract_parameters(self):
        for name, param in self.model.named_parameters():
            if name == 'linear.weight':
                W = param.detach()
            else:
                b = param.detach()

        return W.cpu().numpy(), b.cpu().numpy()

    
    def get_variance(self, y: np.ndarray):
        ey = np.mean(y)
        ey2 = np.mean(np.square(y))
        return ey2 - ey**2

    
    def get_variance_each_feature(self, y: np.ndarray):
        ey = np.mean(y, axis=0)
        ey2 = np.mean(np.square(y), axis=0)
        return ey2 - ey ** 2

    
    def get_variance2(self, y: np.ndarray):
        ey = np.mean(y)
        d = (y - ey) ** 2
        return np.mean(d)

    
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

    
    def test_with_matrix(self, X: np.ndarray, y: np.ndarray):
        W, b = self.extract_parameters()
        diff = y - X @ np.transpose(W) - b
        return np.mean(np.square(diff)), 1 - np.mean(np.square(diff)) / self.get_variance(y)

    
    def test_each_feature(self, X: np.ndarray, y: np.ndarray):
        W, b = self.extract_parameters()
        diff = y - X @ np.transpose(W) - b
        mse = np.mean(np.square(diff), axis=0)
        var = self.get_variance_each_feature(y)
        return mse, 1 - mse / var
    
    
    def test_with_matrix_given(self, X: np.ndarray, y: np.ndarray, W:np.ndarray, b:np.ndarray):
        diff = y - X @ np.transpose(W) - b
        return np.mean(np.square(diff)), 1 - np.mean(np.square(diff)) / self.get_variance(y)

    
    def test_each_feature_given(self, X: np.ndarray, y: np.ndarray, W:np.ndarray, b:np.ndarray):
        diff = y - X @ np.transpose(W) - b
        mse = np.mean(np.square(diff), axis=0)
        var = self.get_variance_each_feature(y)
        return mse, 1 - mse / var

