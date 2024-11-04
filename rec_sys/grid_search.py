import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import data_util as data
from config import ConfigLf as config
import lf_algorithms as lf

# Load and preprocess the data only once
ratings_tf, matrix_u, matrix_v, num_users, num_items = lf.load_data_and_init_factors(config)
train_ds, valid_ds, test_ds = data.split_train_valid_test_tf(ratings_tf, config)

# Define the hyperparameter grid
fixed_learning_rates = jnp.linspace(0.01, 0.1, 5)  # 5 values between 0.01 and 0.1
reg_params = jnp.linspace(0.01, 0.1, 5)  # 5 values between 0.01 and 0.1

best_loss = float('inf')
best_params = (None, None)

# Conduct the grid search
for lr in fixed_learning_rates:
    for reg in reg_params:
        print(f"Training with fixed_learning_rate={lr:.4f} and reg_param={reg:.4f}")

        # Initialize factors anew for each parameter combination
        rng_key_factors, rng_key_r = jax.random.split(jax.random.PRNGKey(config.rng_seed))
        matrix_u, matrix_v = lf.init_latent_factors(num_users, num_items, config.num_factors, rng_key_factors)

        # Run the SGD with L2 regularization for a limited number of epochs
        matrix_u, matrix_v = lf.uv_factorization_vec_reg(matrix_u, matrix_v, train_ds, config, lr, reg)

        # Evaluate the model on the validation dataset
        valid_loss = lf.mse_loss_all_batches(matrix_u, matrix_v, valid_ds, config.batch_size_predict_with_mse)
        valid_loss_mean = jnp.mean(jnp.array(valid_loss))

        print(f"Validation loss: {valid_loss_mean:.6f}")

        # Update best parameters if current loss is lower
        if valid_loss_mean < best_loss:
            best_loss = valid_loss_mean
            best_params = (lr, reg)

print(f"Best parameters found: fixed_learning_rate={best_params[0]:.4f}, reg_param={best_params[1]:.4f}")

# Now train with the best hyperparameters and compare convergence and accuracy
final_lr, final_reg = best_params

# Reinitialize matrices
matrix_u, matrix_v = lf.init_latent_factors(num_users, num_items, config.num_factors, rng_key_factors)

# Train with the best parameters
matrix_u, matrix_v = lf.uv_factorization_vec_reg(matrix_u, matrix_v, train_ds, config, lr, reg)

def show_metrics_and_examples(message, matrix_u, matrix_v):
    print(message)
    mse_all_batches = lf.mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
    print("MSE examples from predict_with_mse on test_ds")
    print(mse_all_batches[:config.num_predictions_to_show])
    print("Prediction examples (pred, target)")
    predictions_and_targets = lf.predict_and_compare(matrix_u, matrix_v, test_ds, config)
    print(predictions_and_targets[:config.num_predictions_to_show])

# Show metrics and examples after training
show_metrics_and_examples("====== After optimization with best hyperparameters =====", matrix_u, matrix_v)

# Compare with original code without regularization
matrix_u_orig, matrix_v_orig = lf.uv_factorization_vec_no_reg(matrix_u, matrix_v, train_ds, valid_ds, config)
show_metrics_and_examples("====== After optimization with original code =====", matrix_u_orig, matrix_v_orig)
