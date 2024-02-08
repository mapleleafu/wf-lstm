import numpy as np
import pandas as pd

def block_bootstrap(df, block_size, n_samples):
    """Generates bootstrap samples of the DataFrame using block sampling."""
    n_blocks = int(len(df) / block_size)
    bootstrap_samples = []
    
    for _ in range(n_samples):
        sample_indices = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, len(df) - block_size + 1)
            indices = list(range(start_idx, start_idx + block_size))
            sample_indices.extend(indices)
        
        sampled_df = df.iloc[sample_indices].reset_index(drop=True)
        bootstrap_samples.append(sampled_df)
    
    return bootstrap_samples

if __name__ == "__main__":
    df_path = "data_tables/processed_EDIRNE-17050Station.csv"
    df = pd.read_csv(df_path)
    
    block_size = 7  # could be adjusted based on data frequency and desired temporal correlation
    n_samples = 5  # Number of bootstrap samples to generate

    # Generate bootstrap samples
    print("Generating bootstrap samples...")
    bootstrap_samples = block_bootstrap(df, block_size, n_samples)

    # Save bootstrap samples to disk
    for i, sample_df in enumerate(bootstrap_samples):
        sample_path = f"bootstrap_samples/bootstrap_sample_{i}.csv"
        sample_df.to_csv(sample_path, index=False)
        print(f"Saved bootstrap sample {i} to {sample_path}")
