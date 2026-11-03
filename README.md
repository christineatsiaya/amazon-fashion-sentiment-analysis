# Amazon Fashion Customer Reviews | NLP & Sentiment Analysis

**Python · Natural Language Processing · Machine Learning · Customer Analytics**

## Project Overview

**How can businesses transform thousands of customer reviews into meaningful insights about customer satisfaction, product quality, and consumer behavior?**

Customer reviews provide valuable information about customer experiences, but analyzing large volumes of unstructured text manually is difficult and time-consuming.

This academic project applies **Natural Language Processing (NLP), text mining, and machine learning** to Amazon Fashion customer reviews to identify patterns, explore recurring feedback themes, and evaluate sentiment classification models.

The project combines exploratory text analysis with predictive modeling to demonstrate how unstructured customer feedback can support business intelligence and data-driven decision-making.

## Project Highlights

| Metric | Result |
|---|---|
| Original Dataset | **400,749 reviews** |
| Cleaned Dataset | **360,159 reviews** |
| Document Similarity Sample | **200 reviews** |
| Clustering Sample | **3,000 reviews** |
| Traditional ML Models | **5 classifiers** |
| Deep Learning Architecture | **LSTM** |
| Best Traditional Model | **Linear SVM** |
| SVM Accuracy | **67%** |
| SVM Weighted F1-Score | **0.62** |

*Results are reported from the saved notebook outputs.*

## Business Problem

Online retailers receive substantial amounts of customer feedback through reviews, ratings, and product evaluations.

Although these reviews contain useful information, businesses face several challenges:

- Identifying recurring complaints across thousands of reviews.
- Understanding common product quality and sizing concerns.
- Distinguishing positive, neutral, and negative feedback.
- Organizing unstructured text into meaningful categories.
- Translating customer feedback into actionable business insights.

This project explores analytical techniques that could help address these challenges.

## Project Objectives

The analysis was organized around three main objectives:

**1. Document Similarity**

Measure textual similarity between customer reviews to identify shared language and related experiences.

**2. Document Clustering**

Use unsupervised learning to explore recurring themes and patterns in customer feedback.

**3. Sentiment Classification**

Train and evaluate machine learning models to predict customer sentiment from review text.

A separate LSTM experiment explores deep learning for binary sentiment classification.

## Tools & Technologies

| Area | Tools and Techniques |
|---|---|
| Programming | Python |
| Data Processing | pandas, NumPy |
| Text Preprocessing | Regular Expressions, Text Normalization |
| Feature Engineering | TF-IDF, Tokenization |
| Similarity Analysis | Cosine Similarity |
| Unsupervised Learning | K-Means, Agglomerative Clustering |
| Supervised Learning | Naive Bayes, SVM, Decision Tree, Random Forest, AdaBoost |
| Deep Learning | TensorFlow, Keras, LSTM |
| Model Evaluation | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook, Google Colab |

## Methodology

### 1. Data Preparation & Cleaning

The original dataset contained **400,749 customer reviews** across 11 columns.

For the analysis, three relevant fields were selected:

- `text` — Customer review content
- `rating` — Product rating
- `target` — Sentiment classification label

The preprocessing workflow included:

- Removing missing review text.
- Removing duplicate reviews.
- Filtering reviews containing fewer than five words.
- Converting text to lowercase.
- Removing URLs, punctuation, and numerical characters.
- Normalizing whitespace.

After filtering, the dataset contained **360,159 reviews**.

This preparation established a more consistent foundation for text analysis and predictive modeling.

### 2. Document Similarity Analysis

**Objective:** Identify reviews with similar textual characteristics.

A random sample of **200 reviews** was selected for document similarity analysis.

The methodology included:

1. Converting cleaned review text into TF-IDF vectors.
2. Computing pairwise cosine similarity scores.
3. Visualizing the similarity matrix using a heatmap.
4. Identifying the ten most similar review pairs.

**Business Insight**

Document similarity can help organizations identify recurring descriptions of customer experiences and potentially repetitive complaints.

It also provides a foundation for organizing related feedback more efficiently.

### 3. Customer Review Clustering

**Objective:** Discover recurring themes within customer reviews using unsupervised machine learning.

A sample of **3,000 reviews** was analyzed using two clustering approaches.

#### K-Means Clustering

TF-IDF features were used to represent review text numerically.

The elbow method was applied to explore cluster selection, with five clusters chosen for the experiment.

The notebook's qualitative interpretation identified themes involving:

| Cluster | Interpreted Theme |
|---|---|
| 0 | Product fit and sizing |
| 1 | Product quality and materials |
| 2 | Shipping and delivery |
| 3 | Positive customer experiences |
| 4 | Returns and complaints |

These themes were interpreted from cluster keywords and are exploratory rather than independently validated categories.

#### Hierarchical Clustering

Agglomerative clustering was also applied to explore relationships between reviews.

The analysis included:

- Hierarchical cluster assignment.
- Dendrogram visualization.
- Dimensionality reduction using Truncated SVD.
- Cluster distribution analysis.
- Comparison with K-Means clustering.

**Key Observation**

Both approaches produced exploratory groupings, but K-Means was preferred in the notebook's interpretation because of its scalability and clearer segmentation.

**Business Insight**

Clustering techniques can help businesses organize customer feedback into recurring topics, making it easier to identify product issues and areas for improvement.

### 4. Sentiment Classification Using Machine Learning

**Objective:** Evaluate how effectively machine learning models classify customer reviews by sentiment.

Five supervised learning algorithms were trained and evaluated:

- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost

The workflow included an **80/20 train-test split**, TF-IDF feature extraction, model training, and evaluation using classification metrics.

#### Model Performance

| Model | Reported Accuracy | Reported Weighted F1-Score |
|---|---:|---:|
| Linear SVM | **67%** | **0.62** |
| Naive Bayes | 65% | 0.59 |
| Random Forest | 63% | 0.61 |

Linear SVM achieved the strongest reported performance among the traditional classification models.

#### Understanding Model Limitations

The SVM model performed better at identifying positive and negative reviews than neutral reviews.

The notebook reported:

- **Negative sentiment recall:** 0.88
- **Positive sentiment recall:** 0.77
- **Neutral sentiment recall:** 0.12

This indicates that the model struggled to distinguish neutral feedback from more clearly positive or negative reviews.

**Business Insight**

A sentiment model with relatively good overall accuracy can still miss important categories of feedback.

For business reporting, class-level performance matters because misclassified neutral reviews could distort customer satisfaction measurements.

### 5. Deep Learning with LSTM

A separate experiment explored sentiment classification using a **Long Short-Term Memory (LSTM)** neural network.

The workflow included:

- Sampling 30,000 reviews before label filtering.
- Mapping negative and positive sentiment to binary labels.
- Tokenizing customer review text.
- Padding text sequences.
- Training an embedding-based LSTM model.
- Applying early stopping.
- Evaluating classification performance.

#### LSTM Results

| Metric | Result |
|---|---:|
| Accuracy | 60.52% |
| Weighted Precision | 0.3662 |
| Weighted Recall | 0.6052 |
| Weighted F1-Score | 0.4563 |

**Evaluation Consideration**

The LSTM experiment used binary sentiment classification, while the traditional models used three sentiment classes.

Because these experiments used different label structures, their results should not be interpreted as a direct head-to-head model comparison.

A consistent dataset, label structure, and evaluation protocol would be required for a controlled comparison.

## Key Findings

### Finding 1: Customer Feedback Contains Recurring Themes

Clustering revealed patterns involving product fit, quality, delivery, satisfaction, and returns.

These themes illustrate how NLP can support customer feedback categorization.

### Finding 2: Traditional Machine Learning Provided a Useful Sentiment Baseline

Linear SVM achieved approximately **67% accuracy** and a weighted F1-score of **0.62** in the three-class classification experiment.

### Finding 3: Neutral Sentiment Was Difficult to Classify

The SVM model achieved only **0.12 recall** for neutral sentiment, demonstrating the importance of evaluating individual classes.

### Finding 4: Text Analytics Can Support Business Decision-Making

The combination of similarity analysis, clustering, and sentiment classification demonstrates how unstructured feedback can be organized into insights relevant to customer experience and product improvement.

## Potential Business Applications

| Business Function | Potential Application |
|---|---|
| Customer Experience | Monitor recurring satisfaction issues |
| Product Management | Identify product quality and sizing concerns |
| Marketing Analytics | Explore customer perceptions and sentiment |
| Quality Assurance | Detect frequently mentioned product problems |
| Business Intelligence | Develop customer feedback reporting |
| Operations | Identify recurring shipping and delivery complaints |

These are potential applications of the techniques explored, not outcomes from a deployed business system.

## Recommendations & Future Improvements

**Improve Sentiment Classification**

Investigate class imbalance, alternative feature representations, and methods to improve neutral sentiment detection.

**Strengthen Model Comparison**

Evaluate traditional and deep learning models using consistent datasets and classification targets.

**Expand Topic Analysis**

Explore additional topic modeling approaches to improve the interpretation of recurring customer concerns.

**Develop Business Intelligence Reporting**

Consider building a dashboard to monitor sentiment distribution, recurring feedback topics, and changes in customer perceptions over time.

**Improve Reproducibility**

Document dataset acquisition, software dependencies, and execution instructions so the analysis can be reproduced more easily.

## Project Limitations

This is an academic analytical project, not a production-deployed sentiment analysis system.

Important limitations include:

- Clustering themes are exploratory and require human interpretation.
- Model performance is specific to the dataset and experimental configuration.
- Neutral sentiment classification remains challenging.
- Traditional and deep learning experiments are not directly comparable.
- No business impact or operational improvements were independently measured.
- The original dataset is not included in this repository.

## Project Notebook

The complete Jupyter notebook contains the preprocessing steps, similarity analysis, clustering experiments, classification models, visualizations, and saved evaluation outputs.

### [View the Complete Python Analysis](./Christine_Analysis_91.ipynb)

**Notebook:** `Christine_Analysis_91.ipynb`

### Running the Notebook

The analysis was developed in Google Colab.

To reproduce the work:

1. Obtain the original Amazon Fashion reviews dataset from its authorized source.
2. Open the notebook in Google Colab or a compatible Jupyter environment.
3. Install the Python packages used in the notebook.
4. Update the dataset path to point to the downloaded CSV file.
5. Run the notebook cells in sequence.

The notebook currently expects the dataset at:

`/content/amazon-fashion-800k+-user-reviews-dataset.csv`

The CSV must contain the required `text`, `rating`, and `target` columns.

## Skills Demonstrated

`Python` · `pandas` · `Data Cleaning` · `Natural Language Processing` · `TF-IDF` · `Cosine Similarity` · `K-Means Clustering` · `Hierarchical Clustering` · `Machine Learning` · `Sentiment Analysis` · `Model Evaluation` · `Data Visualization` · `Customer Analytics`

---

**Project Focus:** Applying NLP and machine learning to transform unstructured customer reviews into meaningful analytical findings and potential business insights.
