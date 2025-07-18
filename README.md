# -MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY : CODTECH IT SOLUTIONS

NAME : Pranjal Manishbhai Patel

INTERN ID : CT06DF1620

DOMAIN : PYTHON PROGRAMMING

DURATION : 6 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

Task Name: Machine Learning Model Implementation
Project Focus: Spam Detection using Naive Bayes Classifier

This project demonstrates how to build a basic yet powerful spam detection model using the Naive Bayes algorithm in Python with the help of the Scikit-learn library. It walks through the typical machine learning pipeline, including data collection, preprocessing, feature extraction, model training, evaluation, and result visualization. The main objective of this task is to classify SMS messages into two categories: spam or ham (non-spam) using a supervised learning approach.

Technologies and Tools Used
Python – The primary programming language used for its rich ecosystem in machine learning and data science.

Pandas – For reading and manipulating the dataset.

NumPy – For numerical operations and array handling.

Scikit-learn (sklearn) – A powerful machine learning library used for preprocessing, vectorization, model training, and evaluation.

Seaborn & Matplotlib – Used for creating the confusion matrix heatmap to visualize the model's performance.

Jupyter Notebook / VS Code – Suitable IDEs for writing, testing, and visualizing ML code.

Implementation Steps
1. Data Collection
The dataset is fetched from an online repository in .tsv format, containing SMS messages labeled as "spam" or "ham." It is loaded into a Pandas DataFrame for processing.

2. Preprocessing
The labels are converted into numerical format (ham = 0, spam = 1) to make them suitable for the machine learning model. This is a crucial step for binary classification.

3. Train-Test Split
The data is split into training and testing sets using an 80-20 ratio. This helps in evaluating the model on unseen data and prevents overfitting.

4. Text Vectorization
Using CountVectorizer, the text data (SMS messages) is transformed into numerical feature vectors. This technique creates a matrix of token counts that machine learning algorithms can understand.

5. Model Training
A Multinomial Naive Bayes classifier is trained on the vectorized training data. This algorithm is well-suited for text classification tasks where features represent discrete counts.

6. Prediction and Evaluation
After training, the model is tested on the unseen test data. Metrics such as accuracy, precision, recall, and F1-score are generated using the classification_report and accuracy_score functions.

7. Visualization
A confusion matrix is created using confusion_matrix and visualized with Seaborn. It clearly shows how many messages were correctly or incorrectly classified.

Applications
Email and SMS Spam Filters: Widely used by Gmail, Outlook, and mobile carriers to detect spam messages.

Social Media Moderation: To automatically filter offensive or unwanted messages.

Customer Service Automation: To distinguish between real customer issues and spam or bot-generated messages.

Cybersecurity: To detect phishing attempts or malicious content.

Academic Use: Ideal for students to understand core ML concepts and apply them to real-world data.

Platform Requirements
OS: Cross-platform (Windows, Linux, macOS)

Environment: Python environment with Jupyter Notebook

Libraries Needed:

pandas

numpy

seaborn

matplotlib

scikit-learn

The entire project runs locally without the need for high-end hardware or GPU support.

Conclusion
This task successfully showcases how machine learning can be used to build practical solutions like spam detectors. By using the Naive Bayes classifier, which is fast and efficient for text data, the model provides high accuracy with minimal training time. This project is ideal for beginners to grasp end-to-end ML workflows and can serve as a stepping stone toward more advanced NLP projects.

Output

<img width="1919" height="1009" alt="Image" src="https://github.com/user-attachments/assets/ffa3a085-e557-4f05-9acb-679f593bdff7" />

<img width="1919" height="382" alt="Image" src="https://github.com/user-attachments/assets/2dbab0ea-56d3-412c-8f38-ad65d429f3aa" />

<img width="1919" height="1017" alt="Image" src="https://github.com/user-attachments/assets/5ce981fd-862e-468b-aaec-e67eb0072868" />
