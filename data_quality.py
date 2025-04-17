import numpy as np

def load_dataset(path='wine.data.txt'):
    data = np.loadtxt(path, delimiter=',')
    data_labels = data[:, 0].astype(int)
    data_points = data[:, 1:]
    return data_labels, data_points

'''
a) Use leave-one-out cross-validation (LOOCV) to estimate the accuracy of the classifier and
also to estimate the 3x3 confusion matrix
'''
def LOOCV(data_labels, data_points):
    number_of_data_points = data_points.shape[0]
    predict_labels = np.zeros(number_of_data_points, dtype=int)
    
    for i in range(number_of_data_points):
        mask = np.arange(number_of_data_points) != i # this will exclude the i-th datapoint
        train_points = data_points[mask] # trainpoints without i-th point
        train_labels = data_labels[mask] # trainlabels without i-th label
        test_point = data_points[i] # i-th datapoint
        
        dists = np.linalg.norm(train_points - test_point, axis=1)
        nn = np.argmin(dists)
        predict_labels[i] = train_labels[nn]
    
    accuracy = np.mean(predict_labels == data_labels)
    conf_matrix = np.zeros((3, 3), int)
    for true, pred in zip(data_labels, predict_labels):
        conf_matrix[true - 1, pred - 1] += 1
    
    return accuracy, conf_matrix

'''
b) Estimate the accuracy of the 1-NN classifier using k-fold cross-validation using 20 different choices
of k that are fairly well spread out across the range 2 to 100. Plot these estimates: put k on the
horizontal axis and accuracy estimate on the vertical axis.
'''
def k_fold_cv_accuracy(data_labels, data_points, k):
    number_of_data_points = data_points.shape[0]
    indices = np.arange(number_of_data_points)
    np.random.seed(0)
    np.random.shuffle(indices)

    fold_sizes = [number_of_data_points // k + (1 if i < number_of_data_points % k else 0) for i in range(k)]
    accuracies = []
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        train_points = data_points[train_idx]
        train_labels = data_labels[train_idx]
        test_points = data_points[test_idx]
        test_labels = data_labels[test_idx]

        preds = np.zeros(test_labels.shape, dtype=int)
        for j, pt in enumerate(test_points):
            dists = np.linalg.norm(train_points - pt, axis=1)
            nn = np.argmin(dists)
            preds[j] = train_labels[nn]

        accuracies.append(np.mean(preds == test_labels))
        start = end

    return np.mean(accuracies)

def plot_kfold_accuracies():
    labels, points = load_dataset('wine.data.txt')
    ks = [int(x) for x in np.linspace(2, 100, 20)]
    accuracies = [k_fold_cv_accuracy(labels, points, k) for k in ks]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ks, accuracies, marker='o', linestyle='-')
    plt.xlabel('Number of folds (k)')
    plt.ylabel('Accuracy estimate')
    plt.title('Accuracy vs k-fold cross-validation')
    plt.grid(True)
    plt.show()

'''
c) The various features in this data set have different ranges. Perhaps it would be better to normalize
them so as to equalize their contributions to the distance function. There are many ways to do
this; one option is to linearly rescale each coordinate so that the values lie in [0, 1] (i.e. the
minimum value on that coordinate maps to 0 and the maximum value maps to 1). Do this, and
then re-estimate the accuracy and confusion matrix using LOOCV. Did the normalization help
performance?
'''
def normalize_features(data_points):
    min_vals = data_points.min(axis=0)
    max_vals = data_points.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    return (data_points - min_vals) / ranges

def main():
    data_labels, data_points = load_dataset('wine.data.txt')

    # --- original data LOOCV ---
    accuracy, conf_matrix = LOOCV(data_labels, data_points)
    print(f'Accuracy of LOOCV using 1-NN classifier with Euclidean distance = {accuracy:.4f}')
    print('   Confusion matrix:')
    print('        Wine1 Wine2 Wine3')
    for i, row in enumerate(conf_matrix):
        print(f'Wine{i+1:<3}', end='')
        for val in row:
            print(f'{val:6d}', end='')
        print()
    print()
    
    # --- normalized data LOOCV ---
    norm_points = normalize_features(data_points)
    norm_acc, norm_cm = LOOCV(data_labels, norm_points)
    print(f'\nAccuracy of LOOCV on normalized features =  {norm_acc:.4f}')
    print('   Confusion matrix (normalized):')
    print('        Wine1 Wine2 Wine3')
    for i, row in enumerate(norm_cm):
        print(f'Wine{i+1:<3}', end='')
        for val in row:
            print(f'{val:6d}', end='')
        print()

if __name__ == '__main__':
    main()
    plot_kfold_accuracies()