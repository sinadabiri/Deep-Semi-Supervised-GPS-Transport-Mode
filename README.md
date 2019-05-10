# Deep-Semi-Supervised-GPS-Transport-Mode
## Summary
This is the code repository for the paper entitled: "Semi-Supervised Deep Learning Approach for Transportation Mode Identification UsingGPS Trajectory Data", submitted to IEEE Transactions on Knowledge and Data Engineering. The paper has been published in IEEE TKDE, which is a high-quality algorithmic journal in data mining. 

The abstract is as follows: 
"Identification of travelers’ transportation modes is a fundamental step for various problems that arise in the domain of transportation such as travel demand analysis, transport planning, and traffic management. In this paper, we aim to identify users’ transportation modes purely based on their GPS trajectories. A majority of studies have proposed mode inference models based on hand-crafted features, which might be vulnerable to traffic and environmental conditions. Furthermore, the classification task in almost
all models have been performed in a supervised fashion while a large amount of unlabeled GPS trajectories has remained unused. Accordingly, in this paper, we propose a deep Semi-Supervised Convolutional Autoencoder (SECA) architecture that can not only automatically extract relevant features from GPS tracks but also exploit useful information in unlabeled data. The SECA integrates a convolutional-deconvolutional autoencoder and a convolutional neural network into a unified framework to concurrently perform supervised and unsupervised learning. The two components are simultaneously trained using both labeled and unlabeled GPS trajectories, which have already been converted into an efficient representation for the convolutional operation. An optimum schedule for varying the balancing parameters between reconstruction and classification errors are also implemented. The performance of the proposed SECA model, the method for converting a raw trajectory into a new representation, the hyperparameter schedule, and the model configuration are evaluated by comparing to several baselines and alternatives for various amounts of labeled and unlabeled data. Our experimental results demonstrate the superiority of the proposed model over the state-of-the-art semi-supervised and supervised methods with respect to metrics such as accuracy and F-measure."

The GeoLife dataset uses in this project is available at: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F

## Code Repository
 All the described data processing and models are implemented within Python programming environment using TensorFlow for deep learning models and scikit-learn for classical supervised algorithms. All experiments are run on a computer with a single GPU.
 
 I divide the codes in four categroies:
 
 1. Data Preprocessing: Files '2-Label-Trajectory-All.py' and'2-Inctance-Creation.py' are preparing trajectory data and matching them with their associated label files to annotate trajectories. Pre-processing steps for removing errors and outliers are also implemented in these codes.
 2. GPS Representation: The file '2-DL-Data-Creation.py' converst the raw GPS trajectories into a novel multi-channle GPS matrix representation, which subsequently is used in deep-learning models.
 3. Deep learning models: Files starting with '2-Conv-Semi-...py' are developing various architectures of deep semi-supervised models. The '2-Conv-Semi-AE+Cls.py' is the code related to my SECA model while the remaining ones are used as baselines. '2-CNN-TF' is related to the supervised (not semi-supervised) CNN model.
 4. Classical baseline methods: The file '2-HandCrafted-Features.py' creates hand-designed features from GPS trajectories and then applies some classical ML methods such as Random forest, KNN, MLP, Decision Tree, and SVM for classifying GPS trajectories. 
 
 ## IMPORTANT NOTE
 **The file '2-Conv-Semi-AE+Cls.py' is a clear example of a complex model than can only be implemented in TensorFlow due to its integrated loss fucntion and the hyperparameters related to the loss function**. I spent one month to code this in Keras but at the end I needed to use TensorFlow. 
 
 **Semi-supervised models can be simply utilized in all types of applications by just changing the settings related to the new data types**

## Contact Information
Please email me at sina@vt.edu if you have any questions. 

