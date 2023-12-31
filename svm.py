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

    #trains the svm with the given data and evaluates the accuracy on the test data
    def train(self, train_data, y_true, test_data, test_y_true, learning_rate=1e-5, epochs=1e5, batch_size=64, eval_freq=1000, save_freq=5000, negative_min_freq =2000, save_dir='weights'):
        num_samples, num_features = train_data.shape
        self.weights = np.random.uniform(low=-1, high=1, size=num_features)
        self.bias = 0.0
        
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()
        
        max_accuracy = 0.0
        accuracy = 0.0
        for epoch in range(int(epochs)):
            epoch_indices = np.arange(num_samples)
            np.random.shuffle(epoch_indices)
            
            train_data_shuffled = train_data[epoch_indices]
            y_true_shuffled = y_true[epoch_indices] #shuffles for a random batch
            
            for i in range(0, num_samples, batch_size): #iterates through the data in batches
                batch_data = train_data_shuffled[i:i+batch_size]
                batch_y_true = y_true_shuffled[i:i+batch_size]
                gradient_weights = 0
                gradient_bias = 0
                for j in range(0, batch_size):
                    if i + j < num_samples:
                        ti = batch_y_true[j] * (np.dot(self.weights, batch_data[j].T) + self.bias) #calculates the gradient
                        if ti <= 1:
                            gradient_weights += batch_y_true[j] * batch_data[j]
                            gradient_bias += batch_y_true[j]
                            

                self.weights += -learning_rate * self.weights + learning_rate * gradient_weights
                self.bias += learning_rate * gradient_bias #updates the weights and bias with the gradient found in the bath
                
            if (epoch+1)% eval_freq == 0: #evaluates the accuracy on the test data on differnt thresholds
                accuracy = self.evaluate_accuracy(test_data, test_y_true)
                print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.4f}")
                self.evaluate_accuracy(test_data, test_y_true, threshold=-0.25)
                self.evaluate_accuracy(test_data, test_y_true, threshold=-0.5)
                self.evaluate_accuracy(test_data, test_y_true, threshold=-0.6)
                self.evaluate_accuracy(test_data, test_y_true, threshold=-0.75)
                self.evaluate_accuracy(test_data, test_y_true, threshold=-0.80)
                self.evaluate_accuracy(test_data, test_y_true, threshold=-0.9)
                print(f'')
                print(f'')
                # breakpoint()
                
            if accuracy > max_accuracy or (epoch+1) % save_freq == 0: #we save weights if accuracy is better than previous best
                max_accuracy = max(accuracy, max_accuracy)              # or if we have reached a save frequency
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                weights_filename = f"weights_{timestamp}_epoch_{epoch + 1}.npz"
                weights_filepath = os.path.join(save_dir, weights_filename)
                self.save_model(weights_filepath)
                print(f"Weights saved at epoch {epoch + 1} with an accuracy of {accuracy:.4f}")
                
            if (epoch+1) % negative_min_freq == 0: #every once in a while, we do negative mining to improve accuracy on false positives
                print(f'Negative mining at epoch {epoch + 1}')
                print(f'{train_data.shape} {y_true.shape}')
                train_data, y_true = self.negative_mine(train_data, y_true)
                print(f'after {train_data.shape} {y_true.shape}')
                # self.negative_mine(train_data, y_true)
                
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Total runtime: {runtime:.2f} seconds")
        
    #does negative mining on the false positives
    def negative_mine(self, train_data, y_true, percentage_false=0.8):
        # breakpoint()
        predictions = self.predict(train_data)
        false_positives_indices = np.where(np.logical_and(predictions > -0.7, y_true == -1))[0] #finds the false positives indices
        
        false_positives = sorted(false_positives_indices, key=lambda x: predictions[x], reverse=False) #sorts the false positives by confidence
        fal = train_data[false_positives[0:int(len(false_positives)*percentage_false)]] #gets the top percentage_false on them
        neg = np.full(len(fal), -2)
        if(len(fal) == 0):
            return
        train_data = np.concatenate((train_data, fal), axis=0) #adds them to the training data
        y_true = np.concatenate((y_true, neg), axis=0)
        return train_data, y_true
        
                
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias #predicts the confidence of the data

    def evaluate_accuracy(self, test_data, y_true, threshold=0.0):
        predictions = self.predict(test_data)
        y_pred = np.where(predictions > threshold, 1, -1)
        tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
        tn = np.sum(np.logical_and(y_pred == -1, y_true == -1))
        fp = np.sum(np.logical_and(y_pred == 1, y_true == -1))
        fn = np.sum(np.logical_and(y_pred == -1, y_true == 1)) #finds the true positives, true negatives, false positives, and false negatives
        print(f'tp {tp} tn {tn} fp {fp} fn {fn} at threshold {threshold}')
        print(f'true accurate {tp/(tp+fn)} true negative {tn/(tn+fp)}') #prints the accuracy
        accuracy = (tp / (tp + fn) + tn / (tn + fp))/2 #finds the accuracy 
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
    # svm.load_model('C:/Users/AaronLo/Documents/cs376/Computer-Vision-Final-Project/weights/weights_20231127_194326_epoch_25000.npz')
    svm.svm_test(test_data, test_y_true)