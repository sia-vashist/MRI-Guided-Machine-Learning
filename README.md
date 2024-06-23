# MRI-Guided-Machine-Learning
Final Year Project | DYPIU 2024

# Project Title: MRI Guided Machine Learning for Early Alzheimers Detection & Forecasting

## Overview
This project investigates the potential of Convolutional Neural Networks (CNNs) in early Alzheimer’s Disease (AD) detection using Magnetic Resonance Imaging (MRI) scans. Starting with an overview of AD and its symptoms like memory loss and body imbalance, the study delves into how the model detects AD, focusing on hippocampal tissue shrinkage and increased white matter in the brain. By evaluating the CNN algorithm, factors such as accuracy, efficiency, cost-effectiveness, reliability, and upgradability are assessed to identify the best AD detection approach. The research employs CNNs in software engineering to develop a precise and early AD detection system using MRI scans, aiming to detect AD early and reduce healthcare costs while emphasizing data quality and robust model development.

---
## Expected Outcome
The expected outcome is a highly accurate CNN model capable of effectively classifying different AD stages, providing better diagnostic tools for patients and healthcare providers. Utilizing a dataset of 5,000 MRI scans from the Alzheimer's Disease Neuroimaging Initiative (ADNI), the study achieves a testing accuracy of 95.17%, showcasing the effectiveness of CNNs in AD classification. This result represents a significant advancement in early detection methods for AD and highlights the transformative potential of machine learning in improving diagnostic practices and patient outcomes.

---
## Experimentation and Validation
The project includes experimentation and validation of CNN models trained for 50, 100, and 200 epochs. Additionally, the misclassified images and confusion matrix are analyzed to gain insights into the model's performance and identify areas for improvement.

- Experimentation:
In our experimentation and validation process, we conducted training runs for varying numbers of epochs to assess the performance of our model in Alzheimer's Disease (AD) detection. For the first set of experiments comprising 50 epochs, our model achieved a commendable accuracy of 91%. Subsequently, we extended the training duration to 100 epochs, resulting in a notable improvement as our model attained a perfect training accuracy of 100%, while the testing accuracy reached an impressive 95%. Encouraged by these promising results, we further extended the training duration to 200 epochs. Remarkably, our model's accuracy surged to an exceptional 99%, demonstrating the robustness and efficacy of our approach in effectively classifying different stages of AD with increasing training epochs. These findings underscore the significance of prolonged training durations in enhancing the model's ability to discern subtle patterns indicative of AD, thus showcasing the potential of our methodology in facilitating early detection and prognosis of the disease.

**50 Epochs** :

<br>
<img width="659" alt="50 epoch" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/37af4c59-553d-4a88-8f15-ff801e2dd289">

---

**100 Epochs**:

<br>
<img width="304" alt="100 epoch" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/7dcf542f-c639-4939-a505-b203cd9e7749">

---

**200 Epochs**:

<br>
<img width="658" alt="200 epoch" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/29d7b6ed-bccb-4c2f-beb7-7fb008da2d66">

---
## Tools and Technologies Used
- Language: Python
- IDE: Anaconda, Jupyter, Spyder
- Libraries/Frameworks: Machine Learning (ML), CNN
- Database: SQL, Valentina Studio
- Image Type: DICOM

## Architechture

![MRI Dataset](https://github.com/siavashist-tech/MRI-Guided-ML/assets/81849824/1fd79242-92b8-48c4-9a9f-1fdc93ccb2f1)

## Graphs and Diagrams
# [flowchart]

![Flowchart](https://github.com/siavashist-tech/MRI-Guided-ML/assets/81849824/640403cd-7054-462e-8966-947688be936b)
<br>
# [Block Diagram]
![Block Diagram](https://github.com/siavashist-tech/MRI-Guided-ML/assets/81849824/c0c75b52-49d8-47d1-a2df-e36702ba13a2)

---
# Results:
**1. Introduction to the CNN Model**
- Model Architecture: SEQUENTIAL
- Training Configuration: Trained with a learning rate scheduler and custom loss function.
  
  <img width="233" alt="SEQUENTIAL" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/cb8f63dc-f73e-4da7-81b2-f0e3b6521f31">

**2. Evaluation Metrics**

<img width="371" alt="Evaluation Metrics" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/fe6d33dd-35be-4149-9932-0cf87dd26e18">

**3. Misclassified Images Analysis** 

<img width="554" alt="misclassified images" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/017940af-5014-47b0-868d-72f71da2b014">

<img width="431" alt="misclassified confusion matrix" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/2177a804-fcf4-45f7-a6e1-ab52ddb87211">

Fined Tuned Model: We first identified misclassified images by comparing the predicted classes from our pre-trained model with the true classes. Then, we preprocessed these images and prepared them with their true labels for fine-tuning. We cloned the pre-trained model and fine-tuned it using the misclassified images, adjusting the model’s parameters to learn task-specific features while retaining previous knowledge. Finally, we optimized the model’s parameters using backpropagation to minimize the loss and improve accuracy.

**4. Correctly Classified Images**

<img width="554" alt="correctly predicted" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/1808bc7c-dddb-4f15-8fe6-3b802863449f">

<img width="431" alt="confusion matrix cnn" src="https://github.com/sia-vashist/MRI-Guided-Machine-Learning/assets/173622122/2912e950-ac96-4a7c-a956-b72c0d0a70f2">

---

# Credits:
Invaluable thanks to the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** for granting us access to their database, without which our project would not have been possible.

We express our deep sense of gratitude to our respected guide and coordinator Dr. Maheshwari Biradar, for her valuable help and guidance. We are thankful for the encouragement she has given us in completing this project successfully.

The report of this major project could not have been accomplished without the periodic suggestions and advice of our project supervisors Dr. Maheshwari Biradar and Dr. Rahul Sharma (Project Coordinator).

We are also grateful to our respected Director, Dr. Bahubali Shiragapur, and Hon’ble Vice Chancellor, DYPIU, Akurdi, Prof. Prabhat Ranjan, for permitting us to utilize all the necessary facilities of the college.

Lastly, thank you to my teammates Siddhant Adsule and Vaishnavi Khade for your coordination and helpful support in the collaborative project.
