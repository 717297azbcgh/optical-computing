# Optical Computing Convolution Data Generation and Reading Guide

This project provides a collection of MATLAB scripts for generating and reading input/output data required for optical computing. It supports two optical computing schemes: the **bit-slicing scheme** and the **conventional scheme**, applicable to both BraTS and MNIST datasets.  
With this tool, image data can be converted into optical computing input format, and the output results can be reconstructed into convolved images for comparative analysis of different schemes.

---

## Data Description

- For the **BraTS** dataset, a randomly selected image sample `output_1480.csv` is used as the input source.  
- For the **MNIST** dataset, the first 200 test images are selected and stored in the `test_samples.csv` file.

---

## Directory Structure

### 1. BraTS Dataset Processing

**Bit-Slicing Scheme:**

- `csv_write.m`  
  Generates optical computing input files from pre-convolution images (e.g., `output_1480.csv`).

- `csv_read.m`  
  Reads the optical computing output results from the bit-slicing scheme and reconstructs the convolved image.

**Conventional Scheme:**

- `csv_write_conv.m`  
  Generates input files required for the conventional scheme.

- `csv_read_conv.m`  
  Reads the optical computing output results and reconstructs the convolved image. Can also be used to analyze error distributions between the two schemes.

---

### 2. MNIST Dataset Processing

**Bit-Slicing Scheme:**

- `csv_write_mnist.m`  
  Generates input files for the bit-slicing scheme using image data from `test_samples.csv`.

- `csv_read_mnist.m`  
  Reads the optical computing output results and reconstructs the convolved image.

**Conventional Scheme:**

- `csv_write_mnist_conv.m`  
  Generates input files required for the conventional scheme for the MNIST dataset.

- `csv_read_mnist_conv.m`  
  Reads the output results and reconstructs the convolved image. Supports error analysis between the two schemes.

---

## Usage Instructions

1. **Select Scheme:** Choose either the bit-slicing scheme or the conventional scheme based on your experiment.
2. **Select Dataset:** Determine whether you are processing BraTS or MNIST data.
3. **Workflow:**
   - Use the appropriate `csv_write*.m` script to generate optical computing input files.
   - Execute the convolution operation using the optical computing platform.
   - Use the corresponding `csv_read*.m` script to read output results and reconstruct the convolved images.

---

## Environment Requirements

- MATLAB R2023a

---

For any questions or assistance, please contact the project maintainer.
