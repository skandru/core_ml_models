import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate a sample image with patterns
def generate_sample_image(size=100):
    """Generate a sample image with some patterns."""
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # Create patterns
    pattern1 = np.sin(np.sqrt(X**2 + Y**2)) / (np.sqrt(X**2 + Y**2) + 1)
    pattern2 = np.cos(0.5 * X) * np.sin(0.5 * Y)
    pattern3 = np.exp(-(X**2 + Y**2) / 100)
    
    # Combine patterns
    image = pattern1 + pattern2 + pattern3
    
    # Normalize to 0-255 range
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    return image

# Create sample image
image_size = 100
original_image = generate_sample_image(image_size)

print("Step 1: Original Image Information")
print(f"Image shape: {original_image.shape}")
print(f"Image size: {original_image.size} pixels")
print(f"Memory usage: {original_image.nbytes / 1024:.2f} KB")

# Reshape image for PCA
X = original_image.reshape(-1, image_size)

# Try different compression ratios
compression_ratios = [0.9, 0.5, 0.2, 0.1]

print("\nStep 2: Compression Analysis")
for ratio in compression_ratios:
    # Calculate number of components to keep
    n_components = int(ratio * min(X.shape))
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_compressed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_compressed)
    
    # Reshape back to image
    reconstructed_image = X_reconstructed.reshape(image_size, image_size)
    
    # Calculate compression metrics
    original_size = original_image.nbytes
    compressed_size = X_compressed.nbytes + pca.components_.nbytes
    compression_ratio = compressed_size / original_size
    
    # Calculate reconstruction error
    mse = np.mean((original_image - reconstructed_image) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    
    print(f"\nCompression Ratio: {ratio:.1f}")
    print(f"Number of components: {n_components}")
    print(f"Original size: {original_size / 1024:.2f} KB")
    print(f"Compressed size: {compressed_size / 1024:.2f} KB")
    print(f"Actual compression ratio: {compression_ratio:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")

# Function to visualize compression results
def plot_compression_results(original, n_components_list):
    """Plot original image and compressed versions."""
    n_plots = len(n_components_list) + 1
    plt.figure(figsize=(15, 3))
    
    # Plot original
    plt.subplot(1, n_plots, 1)
    plt.imshow(original, cmap='viridis')
    plt.title('Original')
    plt.axis('off')
    
    # Plot compressed versions
    for i, n_comp in enumerate(n_components_list, 1):
        pca = PCA(n_components=n_comp)
        X_compressed = pca.fit_transform(original.reshape(-1, image_size))
        X_reconstructed = pca.inverse_transform(X_compressed)
        reconstructed = X_reconstructed.reshape(image_size, image_size)
        
        plt.subplot(1, n_plots, i + 1)
        plt.imshow(reconstructed, cmap='viridis')
        plt.title(f'{n_comp} components\n({n_comp/image_size:.1%})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Analyze compression performance
print("\nStep 3: Detailed PCA Analysis")
pca_full = PCA()
pca_full.fit(X)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components needed for different variance thresholds
variance_thresholds = [0.8, 0.9, 0.95, 0.99]
components_needed = []
for threshold in variance_thresholds:
    n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
    components_needed.append(n_components)
    print(f"\nComponents needed for {threshold:.0%} variance: {n_components}")
    print(f"Compression ratio: {n_components/image_size:.2%}")

# Test compression and reconstruction with optimal components
print("\nStep 4: Optimal Compression")
optimal_components = components_needed[1]  # Using 90% variance threshold
pca_optimal = PCA(n_components=optimal_components)
X_compressed = pca_optimal.fit_transform(X)
X_reconstructed = pca_optimal.inverse_transform(X_compressed)
reconstructed_image = X_reconstructed.reshape(image_size, image_size)

# Calculate final metrics
compression_ratio = (X_compressed.nbytes + pca_optimal.components_.nbytes) / original_image.nbytes
mse = np.mean((original_image - reconstructed_image) ** 2)
psnr = 20 * np.log10(255 / np.sqrt(mse))

print(f"\nFinal Results with {optimal_components} components:")
print(f"Compression ratio: {compression_ratio:.2f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"Explained variance: {np.sum(pca_optimal.explained_variance_ratio_):.3f}")

# Visualize results with different numbers of components
components_to_show = [5, 10, 20, 50]
plot_compression_results(original_image, components_to_show)

# Save compressed data
np.savez('compressed_image.npz', 
         compressed_data=X_compressed, 
         components=pca_optimal.components_,
         mean=pca_optimal.mean_)

print("\nStep 5: Loading and Reconstructing")
# Load compressed data
loaded_data = np.load('compressed_image.npz')
X_compressed = loaded_data['compressed_data']
components = loaded_data['components']
mean = loaded_data['mean']

# Reconstruct image
X_reconstructed = np.dot(X_compressed, components) + mean
reconstructed_image = X_reconstructed.reshape(image_size, image_size)

print("Compression and reconstruction completed!")
print(f"Original image size: {original_image.nbytes / 1024:.2f} KB")
print(f"Compressed data size: {(X_compressed.nbytes + components.nbytes + mean.nbytes) / 1024:.2f} KB")