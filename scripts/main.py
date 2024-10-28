from scripts.data_processing import load_data, save_data, split_data, create_technical_indicators, prepare_features_and_target
from scripts.modeling import train_random_forest, evaluate_model, rf_objective
import optuna

def main():
    # Load dataset
    df = load_data('../data/raw/1INCH-BTC.csv')
    
    # Preprocess data and create technical indicators
    df = create_technical_indicators(df)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_random_forest(X_train, y_train)

    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f'MSE: {mse}, R2: {r2}')
    
    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: rf_objective(trial, X, y), n_trials=25)

if __name__ == '__main__':
    main()
