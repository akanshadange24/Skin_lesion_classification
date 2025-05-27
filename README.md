Skin cancer is among the most prevalent forms of cancer globally, and early
detection plays a critical role in improving patient outcomes. This thesis presents
a comprehensive deep learning-based framework for the automated classification
of dermoscopic skin lesion images using the HAM10000 dataset. The proposed
methodology involves multiple stages, including image preprocessing, data segmentation, and the application of advanced convolutional neural network (CNN)
architectures through transfer learning models. The performance of seven cuttingedge pretrained models—ResNet50, MobileNetV2, VGG19, DenseNet121, InceptionV3, Xception, and EfficientNetB0—was assessed and refined separately. Weighted
loss functions and class-balancing techniques were used to address the dataset’s
intrinsic class imbalance problem. Additionally, a performance-based weighted
soft voting mechanism was used to integrate all base learners to create a unique
Uncertainty-Guided Ensemble Model. To increase prediction accuracy and model
calibration, uncertainty quantification methods including temperature scaling and
confidence thresholding were used. Compared to individual models, the ensemble model performed better, outperforming the best standalone model (EfficientNetB0) by 1.68% in accuracy and 1.95% in F1 score, with an accuracy of 85.42%
and an F1 score of 85.93%. A wide range of metrics, including accuracy, recall,
specificity, F1-score, and AUC-ROC, were used for evaluation, both overall and for
each class. This study opens the door for practical implementation in computeraided dermatological diagnosis systems by proving the effectiveness of integrating
ensemble learning, transfer learning, and uncertainty modelling for accurate and
dependable skin lesion classification.# Skin_lesion_classification
