# parallelANN
Fruit Disease Identification Using Parallelized ANN 

This project presents a comprehensive method for developing a fruit disease recognition model using parallelized Artificial Neural Networks (ANNs), specifically focusing on detecting diseases in papaya. Leveraging the exceptional performance of ANNs in image classification, the model aims to accurately identify various fruit diseases and differentiate between diseased and healthy fruits.

This project provides a systematic workflow for developing and deploying a fruit disease recognition model focused on identifying diseases in papaya. It begins with data collection, where high-quality images of healthy and diseased papaya fruits are organized into training and testing directories. Next, data preprocessing techniques, such as augmentation (rotation, zoom, width/height shifts), are applied to enhance the dataset and improve model robustness. 
An Artificial neural network (ANN) is then created using TensorFlow and Keras, featuring multiple convolutional layers, max pooling, dropout for regularization, and dense layers for classification. The model is trained with varying configurations, including serial and parallel execution, to measure performance. After training, the models are evaluated on a test dataset to assess accuracy and generalization, with a comparison of execution times between the two approaches. 

For deployment, a Flask web application is built to allow users to upload papaya images for disease prediction, utilizing both models to display results. Finally, the application presents prediction outcomes along with execution times, providing visual feedback on the performance differences, thus integrating advanced machine learning techniques with user-friendly deployment for effective disease detection and management of papaya health in agricultural practices.

