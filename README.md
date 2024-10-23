# Sleep Posture and Subject Classification using multitask convolutional neural network (CNN)

### Project Overview
This project develops a multitask convolutional neural network (CNN) model for classifying sleep postures and identifying subjects with high accuracy, demonstrating its robustness and potential application in clinical settings through comparative experiments and cross-validation techniques.

### Project Members
- Fairouz Baz Radwan
- Sofia Pope Trogu

### Timeline
July 2024

### Key Features
- **Multitask CNN Model**: Used a multitask CNN model and employed k-fold cross-validation to identify the optimal lambda value, ensuring a balance between posture clas- sification accuracy and subject-specific loss.
- **Architecture Comparison**: Evaluated base CNN vs. smaller CNN models for effectiveness and efficiency.
- **Validation**: Tested generalizability with Leave-One-Subject-Out (LOSO) and k-fold cross-validation.
- **K-Means Clustering**: Applied KMeans post dimensionality reduction to CNN feature representations, demonstrating the versatility of our model for related tasks.

### Datasets
This study used the PmatData public dataset, which includes pressure data from 13 participants (ages 19-34) in 17 different postures, collected at 1 Hz using Vista Medical FSA SoftFlex 2048. Each participant's data consists of 64x32 pressure matrices, with sensors reporting values between 0-1000, highlighting three main postures: supine, right side, and left side. Key data preprocessing steps include:  
1. Applied a 3x3 median spatial filter to each reshaped pressure matrix to reduce noise while preserving edges.
2. Normalized pressure data from [0-1000] to [0-255] for machine learning compatibility.
3. Removed the first and last three frames of each file due to corrupted images.

### CNN Architecture
The base CNN model processes 64x32x1 pressure matrices through four convolutional layers (32, 64, 128, 256 filters), each followed by batch normalization, Leaky ReLU activation, dropout, and max-pooling, culminating in global max-pooling and two dense layers for subject and posture classification. Two smaller CNN models (medium and small) with reduced filters and neurons were also implemented to compare classification accuracy and computation time.

### Libraries Used
- `matplotlib.pyplot`, `seaborn`, `scipy`, `PIL` : Data visualization.
- `numpy`, `pandas`, `os`, `time`: Data manipulation and time handling.  
- `tensorflow`: CNN architecture.
- `sklearn` packages: Data preprocessing, model evaluation, clustering, dimensionality reduction, and optimization.

### Visualizations
This project includes plots of confusion matrices illustrating the test accuracy of the various CNN models, as well as visualizations of dimensionality reduction of features and Kmeans clustering methods.
