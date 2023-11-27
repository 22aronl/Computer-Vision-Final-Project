import numpy as np
import time
import os

from process_datasets import read_image_set, read_train_set, read_test_set

#linear svm
class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.weights = None
        self.bias = None
    
    def hinge_loss(self, y_true, y_pred):
        reg = 0.5 * (self.weights * self.weights)

        for i in range(y_pred.shape[0]):
            # Optimization term
            opt_term = y_true[i] * y_pred[i]

            # calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]
        # return np.maximum(0, 1 - y_true * y_pred)


    def train(self, train_data, y_true, test_data, test_y_true, learning_rate=1e-5, epochs=1e5, batch_size=64, eval_freq=5000, save_freq=30000, save_dir='weights'):
        num_samples, num_features = train_data.shape
        self.weights = np.random.uniform(low=-1, high=1, size=num_features)
        self.bias = 0.0
        
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()
        
        for epoch in range(int(epochs)):
            epoch_indices = np.arange(num_samples)
            np.random.shuffle(epoch_indices)
            
            train_data_shuffled = train_data[epoch_indices]
            y_true_shuffled = y_true[epoch_indices]
            
            # breakpoint()
            for i in range(0, num_samples, batch_size):
                batch_data = train_data_shuffled[i:i+batch_size]
                batch_y_true = y_true_shuffled[i:i+batch_size]
                gradient_weights = 0
                gradient_bias = 0
                for j in range(0, batch_size):
                    if i + j < num_samples:
                        ti = batch_y_true[j] * (np.dot(self.weights, batch_data[j].T) + self.bias)
                        if ti > 0:
                            pass
                        else:
                            gradient_weights += batch_y_true[j] * batch_data[j]
                            gradient_bias += batch_y_true[j]
                #batch_y_ture 1x64
                #batch_data 64 x 800
                self.weights += -learning_rate * self.weights + learning_rate * gradient_weights
                self.bias += learning_rate * gradient_bias
                # self.weights += gradient_weights
                
                # self.bias += learning_rate * gradient_bias
                
            if (epoch+1)% eval_freq == 0:
                accuracy = self.evaluate_accuracy(test_data, test_y_true)
                print(f"Epoch {epoch}/{epochs}, Test Accuracy: {accuracy:.4f}")
                # breakpoint()
                
            if (epoch+1) % save_freq == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                weights_filename = f"weights_{timestamp}_epoch_{epoch + 1}.npz"
                weights_filepath = os.path.join(save_dir, weights_filename)
                self.save_model(weights_filepath)
                print(f"Weights saved at epoch {epoch + 1}")
                
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Total runtime: {runtime:.2f} seconds")
                
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate_accuracy(self, test_data, y_true):
        predictions = self.predict(test_data)
        y_pred = np.where(predictions > 0, 1, -1)
        # breakpoint()
        accuracy = np.mean(y_pred == y_true)
        return accuracy
    
    def svm_test(self, test_data, y_true):
        accuracy = self.evaluate_accuracy(test_data, y_true)
        print(f"Test Accuracy: {accuracy:.4f}")
                
                
    def save_model(self, file_path):
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load_model(self, file_path):
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']
            
    
if __name__ == '__main__':  
    svm = SVM()
    
    train_data, y_true = read_train_set()#read_image_set([1, 2])
    test_data, test_y_true = read_test_set()#read_image_set([5])
    
    print(f'Train data shape: {train_data.shape} {(test_data.shape)}')
    
    svm.train(train_data, y_true, test_data, test_y_true)
    svm.save_model('final_weights.npz')
    svm.load_model('final_weights.npz')
    svm.svm_test(test_data, test_y_true)