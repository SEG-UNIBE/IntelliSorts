"""
Model Training
"""
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

tf.random.set_seed(42)
np.random.seed(42)


def grid_search(model_name, train_input, train_output, param_grid, test_size=0.3, val_size=0.5, random_state=42):
    """
    Performs a manual grid search over neural network hyperparameters, trains models,
    evaluates validation accuracy, selects the best model, and saves training artifacts.

    This function builds and evaluates fully-connected feed-forward neural networks
    for multi-class classification using a manually specified hyperparameter grid.
    It splits the data into train/validation/test subsets, standardizes input features,
    encodes categorical labels, and trains one model per parameter combination. The
    model with the highest validation accuracy is selected, evaluated on the test set,
    and saved to disk along with preprocessing objects.

    Parameters
    ----------
    model_name : str
        Base name used when saving the resulting model and preprocessing files. 
        Output files include:
            - <model_name>.keras              (best-performing neural network model)
            - <model_name>_grid_results.pkl   (grid search results DataFrame)
            - <model_name>_scaler.pkl         (StandardScaler fitted on training data)
            - <model_name>_label_encoder.pkl  (LabelEncoder for target classes)
    
    train_input : pandas.DataFrame
        Input feature matrix used for model training. Typically includes
        presortedness metrics such as 'Inversions', 'Deletions', 'Runs', 'Dis'.
    
    train_output : pandas.Series or array-like
        Target labels corresponding to each training example (e.g., algorithm names).
    
    param_grid : dict
        Dictionary specifying the hyperparameter search space. Keys correspond to
        model parameters and values are lists of values to test. Expected keys:
            - 'batch_size' : list[int]
            - 'epochs' : list[int]
            - 'layers' : list[int]
            - 'layersize' : list[int]
    
    test_size : float, optional (default=0.3)
        Proportion of the dataset to hold out for validation and test splitting.
    
    val_size : float, optional (default=0.5)
        Fraction of the held-out set (from test_size) to allocate for validation.
        The remaining fraction is used as the test set.
    
    random_state : int, optional (default=42)
        Random seed for reproducible dataset splitting.
    
    Returns
    -------
    test_indices : pandas.Index
        Indices of the samples belonging to the test set.
    
    test_true_algorithms : numpy.ndarray
        True algorithm labels for the test set.
    
    test_predicted_algorithms : numpy.ndarray
        Predicted algorithm labels for the test set produced by the best-performing model.
    
    Notes
    -----
    - Uses TensorFlow/Keras sequential models with ReLU activation in hidden layers
      and a softmax output layer for multi-class classification.
    - Best model selection is based on validation accuracy during grid search.
    - All trained models are ephemeral except for the best-performing one, which is saved.
    - The StandardScaler and LabelEncoder are persisted for future preprocessing consistency.
    - Returns decoded predictions for convenience in evaluation or reporting.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(train_output)
    
    X_train, X_split, y_train, y_split = train_test_split(train_input, y_encoded, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, test_size=val_size, random_state=random_state)
    
    train_indices = X_train.index
    test_indices = X_test.index
    
    # standardizing the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    # neural network
    
    def create_model(input_dim, layers, layersize):
        model = Sequential()
        model.add(Dense(layersize, input_dim=input_dim, activation='relu'))
        for i in range(layers):
            model.add(Dense(layersize, activation='relu'))
            
        model.add(Dense(len(label_encoder.classes_), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
        return model
    
    best_accuracy = 0
    best_params = None
    
    grid_results = []
    
    for params in ParameterGrid(param_grid):
        print("Training model with params:", params)
        input_dim = train_input.shape[1]
        model = create_model(input_dim = input_dim, layers = params['layers'], layersize = params['layersize'])
        model.fit(X_train_scaled, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)
        y_pred = model.predict(X_val_scaled)
        accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)[1]
        grid_results.append({
            'Layers': params['layers'],
            'Layer Size': params['layersize'],
            'Accuracy': accuracy
        })
        print("Accuracy:", accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model
    
    df_grid_results = pd.DataFrame(grid_results)
    df_grid_results_sorted = df_grid_results.sort_values(by='Accuracy', ascending=False)
    print(df_grid_results_sorted)
    
    print("Best validation parameters:", best_params)
    print("Best validation accuracy:", best_accuracy)
    print("\n")

    # save the grid results dictionary to a file
    with open(f'{model_name}_grid_results.pkl', 'wb') as f:
        pickle.dump(df_grid_results, f)

    # save the best model
    best_model.save(f"{model_name}.keras")

    # save the scaler and label encoder
    with open(f"{model_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{model_name}_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("Model, scaler, and label encoder have been saved successfully.")

    accuracy, test_true_algorithms, test_predicted_algorithms = evaluate_model(best_model, label_encoder, X_test_scaled, y_test)
    return best_model, scaler, label_encoder, accuracy, test_indices, test_true_algorithms, test_predicted_algorithms


def evaluate_model(best_model, label_encoder, X_test_scaled, y_test):
    y_pred = best_model.predict(X_test_scaled)
    accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    print("Test Accuracy:", accuracy)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_algorithms = label_encoder.inverse_transform(y_pred_classes)
    true_algorithms = label_encoder.inverse_transform(y_test)
    comparison_df = pd.DataFrame({'Predicted Algorithm': predicted_algorithms, 'True Algorithm': true_algorithms})
    
    actual_counts = comparison_df["True Algorithm"].value_counts()
    predicted_counts = comparison_df["Predicted Algorithm"].value_counts()
    
    summary_table = pd.DataFrame({
        "Optimal Count": actual_counts,
        "Predicted Count": predicted_counts
    })
    
    print()
    print(summary_table)
    
    correct_merge_count = ((comparison_df["True Algorithm"] == "Mergesort") & (comparison_df["Predicted Algorithm"] == "Mergesort")).sum()
    correct_timsort_count = ((comparison_df["True Algorithm"] == "Timsort") & (comparison_df["Predicted Algorithm"] == "Timsort")).sum()
    correct_insertion_count = ((comparison_df["True Algorithm"] == "Insertionsort") & (comparison_df["Predicted Algorithm"] == "Insertionsort")).sum()
    correct_quick_sort_count = ((comparison_df["True Algorithm"] == "Quicksort") & (comparison_df["Predicted Algorithm"] == "Quicksort")).sum()
    correct_introsort_count = ((comparison_df["True Algorithm"] == "Introsort") & (comparison_df["Predicted Algorithm"] == "Introsort")).sum()
    
    print()
    print("Correctly predicted Insertionsort:", correct_insertion_count)
    print("Correctly predicted Timsort:", correct_timsort_count)
    print("Correctly predicted Mergesort:", correct_merge_count)
    print("Correctly predicted Quicksort:", correct_quick_sort_count)
    print("Correctly predicted Introsort:", correct_introsort_count)

    return accuracy, true_algorithms, predicted_algorithms
    