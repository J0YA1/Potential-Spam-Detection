**MODEL DESIGN DOCUMENT**

**Predict Potential Spammers**

*Capstone Project 1 — Binary Classification for Fraudulent Job-Offer Detection*

Document Version: 1.0

Project Duration: 21 Days

Language: Python  |  Frameworks: Scikit-Learn, XGBoost, LightGBM

Prepared by: Joyal

Data Source: Kaggle — Fiverr Predict Potential Spammers Dataset

# <a name="_toc234698605"></a>**Table of Contents**
[**Table of Contents**	1](#_toc234698605)

[**1. Executive Summary**	1](#_toc234698606)

[**2. Objective**	1](#_toc234698607)

[**2.1 Project Parameters**	1](#_toc234698608)

[**3. Business Understanding**	1](#_toc234698609)

[**3.1 Goal**	1](#_toc234698610)

[**3.2 Business Problem**	1](#_toc234698611)

[**3.3 Success Criteria**	1](#_toc234698612)

[**3.4 Deliverables**	1](#_toc234698613)

[**3.5 Stakeholders & Impact**	1](#_toc234698614)

[**4. Data Collection**	1](#_toc234698615)

[**4.1 Dataset Overview**	1](#_toc234698616)

[**4.2 Initial Inspection**	1](#_toc234698617)

[**4.3 Data Handling Decision**	1](#_toc234698618)

[**5. Exploratory Data Analysis**	1](#_toc234698619)

[**5.1 Missing Value Treatment**	1](#_toc234698620)

[**5.2 Constant / Zero-Variance Features**	1](#_toc234698621)

[**5.3 Class Imbalance**	1](#_toc234698622)

[**5.4 Summary of EDA Outcomes**	1](#_toc234698623)

[**6. Environment & Tooling**	1](#_toc234698624)

[**6.1 Platform Selection**	1](#_toc234698625)

[**6.2 Compute Considerations**	1](#_toc234698626)

[**6.3 Toolchain Summary**	1](#_toc234698627)

[**7. Feature Engineering**	1](#_toc234698628)

[**7.1 Skewness Correction**	1](#_toc234698629)

[**7.2 Feature Scaling**	1](#_toc234698630)

[**7.3 Correlation Analysis & Dimensionality Reduction**	1](#_toc234698631)

[**7.4 Multicollinearity Check (VIF)**	1](#_toc234698632)

[**8. Model Development**	1](#_toc234698633)

[**8.1 Train/Test Split Strategy**	1](#_toc234698634)

[**8.2 Algorithms Evaluated**	1](#_toc234698635)

[**8.3 Initial Ranking by Recall**	1](#_toc234698636)

[**8.4 Cross-Validation Strategy**	1](#_toc234698637)

[**9. Model Evaluation**	1](#_toc234698638)

[**9.1 Baseline Result**	1](#_toc234698639)

[**9.2 Stratified K-Fold Result**	1](#_toc234698640)

[**9.3 Hyperparameter Tuning**	1](#_toc234698641)

[**9.4 Final Model Summary**	1](#_toc234698642)

[**9.5 Interpretation**	1](#_toc234698643)

[**10. Model Deployment**	1](#_toc234698644)

[**10.1 Deployment Architecture**	1](#_toc234698645)

[**10.2 Packaging & Reproducibility**	1](#_toc234698646)

[**10.3 Operational Considerations**	1](#_toc234698647)

[**11. Model Monitoring & Maintenance**	1](#_toc234698648)

[**11.1 Performance Monitoring**	1](#_toc234698649)

[**11.2 Data Drift Monitoring**	1](#_toc234698650)

[**11.3 Retraining Cadence**	1](#_toc234698651)

[**12. Risks, Assumptions & Limitations**	1](#_toc234698652)

[**12.1 Assumptions**	1](#_toc234698653)

[**12.2 Known Limitations**	1](#_toc234698654)

[**12.3 Risk Mitigation**	1](#_toc234698655)

[**13. Conclusion & Key Learnings**	1](#_toc234698656)

[**13.1 Key Learnings**	1](#_toc234698657)

[**14. Future Enhancements**	1](#_toc234698658)

[**15. References**	1](#_toc234698659)





Predict Potential Spammers — MDD v1.0
# <a name="_toc234698606"></a>**1. Executive Summary**
Freelance and gig-work platforms increasingly face a specific and costly abuse pattern: bad actors post fake “job offers” that carry malware-laced attachments disguised as project briefs, or that lure freelancers into paying bidding fees for work that never existed. Victims lose money, time, and trust in the platform. This Model Design Document (MDD) describes the end-to-end design, build, and evaluation of a binary classification model that flags a user as a potential spammer based on 51 anonymized behavioral and profile attributes (X1–X51).

The final model, a hyperparameter-tuned LightGBM classifier trained on a stratified 5-fold cross-validation scheme, achieved 99% overall accuracy and a recall of 67–71% on the minority (spammer) class, comfortably exceeding the project's success threshold of recall > 60%. This document expands the original milestone notes into a complete, submission-ready MDD covering business understanding, data collection, exploratory analysis, feature engineering, model development, evaluation, deployment via Streamlit, and ongoing monitoring.
# <a name="_toc234698607"></a>**2. Objective**
Find a well-fitted machine learning algorithm capable of predicting, with high reliability, whether a platform user is a potential spammer, using anonymized numeric features that protect the underlying raw data while still preserving its predictive signal.
## <a name="_toc234698608"></a>**2.1 Project Parameters**

|**Parameter**|**Detail**|
| :- | :- |
|Project Duration|21 days|
|Model Type|Binary Classification|
|Language|Python 3.12.0|
|Frameworks|Scikit-Learn, XGBoost, CatBoost, LightGBM|
|Deployment|Streamlit|
|Document Version|1\.0|


# <a name="_toc234698609"></a>**3. Business Understanding**
## <a name="_toc234698610"></a>**3.1 Goal**
Develop a machine learning model to predict whether a given user is a potential spammer or not, based on 51 anonymized columns (X1–X51) whose real-world meaning has been encrypted/masked to protect proprietary and personal data.
## <a name="_toc234698611"></a>**3.2 Business Problem**
A growing volume of malicious activity on freelance marketplaces involves job postings that impersonate legitimate client work. These postings often carry attachments containing malware, or exist purely to extract bidding fees or personal information from freelancers under the pretense of a real project. The financial and reputational damage compounds: freelancers lose earnings and fees, and the platform loses user trust. The core business need is therefore a reliable, automated way to flag potential spammers before they can victimize other users.
## <a name="_toc234698612"></a>**3.3 Success Criteria**
- Achieve the best possible F1 score given the class imbalance in the dataset.
- Achieve a recall score greater than 60% on the spammer (positive) class, since missing an actual spammer (a false negative) is more costly to the business than a false alarm.
## <a name="_toc234698613"></a>**3.4 Deliverables**
- A trained, serialized classification model.
- An end-to-end prediction pipeline (pre-processing + inference).
- Complete project documentation (this MDD).
- A Streamlit application exposing the model for interactive scoring.
## <a name="_toc234698614"></a>**3.5 Stakeholders & Impact**
Primary stakeholders include the Trust & Safety team responsible for platform integrity, freelancers who bear the direct financial risk of spam job postings, and platform operations teams who would consume model output to trigger manual review, account suspension, or warning workflows. A false negative (an undetected spammer) exposes freelancers to fraud; a false positive (a legitimate user wrongly flagged) creates friction and potential churn — which is why both recall and precision are tracked, with recall prioritized per the success criteria.


# <a name="_toc234698615"></a>**4. Data Collection**
## <a name="_toc234698616"></a>**4.1 Dataset Overview**

|**Attribute**|**Value**|
| :- | :- |
|Source|Kaggle — Fiverr Predict Potential Spammers|
|Samples|458,798|
|Features|53 (X1–X51, plus identifier and target)|
|Target|Label (Spammer: Yes/No)|
## <a name="_toc234698617"></a>**4.2 Initial Inspection**

|**Check**|**Finding**|
| :- | :- |
|Missing Values|6 missing values found, isolated to column X13|
|Duplicate Rows|No duplicate rows detected|
|Target Distribution|Imbalanced — approximately 97.6% “No” vs. 2.4% “Yes”|
|Data Types|int64, float64 (fully numeric / anonymized)|
## <a name="_toc234698618"></a>**4.3 Data Handling Decision**
The decision was made to proceed with the raw dataset without aggressive cleaning, since some apparently “noisy” values could carry genuine predictive signal that would be lost through over-cleaning. As a direct consequence of this decision, Logistic Regression and KNN Classifier were provisionally excluded from the primary modeling shortlist, since both are sensitive to unscaled, skewed, or noisy features and to the sharp class imbalance present in the data. They were, however, retained for later exploratory comparison once feature engineering was complete, to confirm whether preprocessing narrowed the performance gap.


# <a name="_toc234698619"></a>**5. Exploratory Data Analysis**
## <a name="_toc234698620"></a>**5.1 Missing Value Treatment**
Only column X13 contained missing values, and only 6 records were affected out of 458,798 — a negligible fraction. Median imputation was selected over the alternatives for the following reasons:

- Mean imputation was avoided because the underlying distribution is imbalanced and prone to outliers, which would bias the mean upward or downward relative to the “typical” value.
- Mode imputation was avoided since X13 is a continuous-like numeric attribute, not categorical, so a single repeating mode would not represent the data well.
- KNN imputation was avoided due to its computational cost on a dataset of this size (458K+ rows), which would have added significant runtime without a proportional accuracy benefit for only 6 missing cells.
- Row deletion was avoided because every other attribute was fully populated; discarding rows would have destroyed otherwise-complete, valuable records for the sake of one sparsely-missing column.
## <a name="_toc234698621"></a>**5.2 Constant / Zero-Variance Features**
After imputation, a unique-value scan was run across all attributes. Seven columns — X27, X29, X30, X33, X46, X47, and X48 — were found to contain a single unique value (0) across the entire dataset. A feature with zero variance carries no discriminative information for a classifier and cannot be feature-engineered into something useful (this would only be possible if such a column had at least two distinct values to work with). These seven columns were therefore dropped, reducing the feature space from 53 to 46 usable attributes.
## <a name="_toc234698622"></a>**5.3 Class Imbalance**
The target label is heavily imbalanced, with spammers representing only ~2.4% of all records. This imbalance directly informed downstream decisions: the use of stratified train/test splitting, stratified K-Fold cross-validation, and the prioritization of recall (rather than raw accuracy) as the primary success metric, since a naive model could achieve 97%+ accuracy simply by predicting “not spammer” for every record.
## <a name="_toc234698623"></a>**5.4 Summary of EDA Outcomes**
- 46 usable attributes retained after removing 7 zero-variance columns.
- 6 missing values in X13 resolved via median imputation.
- No duplicate records found.
- Significant class imbalance (~97.6% / ~2.4%) confirmed and flagged for the modeling stage.


# <a name="_toc234698624"></a>**6. Environment & Tooling**
## <a name="_toc234698625"></a>**6.1 Platform Selection**
Development was carried out in Jupyter Notebooks running Python 3.12.0, chosen for its interactive, cell-by-cell workflow which suits iterative EDA and feature engineering. A standard environment definition was captured in requirements.txt so that the project can be reproduced consistently across machines.
## <a name="_toc234698626"></a>**6.2 Compute Considerations**
Training was run on native CPU rather than GPU, since GPU acceleration for the chosen libraries (Scikit-Learn, XGBoost, CatBoost, LightGBM) was not reliably supported on the Windows development environment used for this project. This constraint influenced algorithm selection and runtime expectations, particularly favoring LightGBM, which is well known for efficient CPU-based training on large tabular datasets.
## <a name="_toc234698627"></a>**6.3 Toolchain Summary**

|**Layer**|**Tool / Library**|
| :- | :- |
|Notebook / IDE|Jupyter Notebook|
|Data Handling|Pandas, NumPy|
|Preprocessing|Scikit-Learn (PowerTransformer, MinMaxScaler, PCA)|
|Modeling|Scikit-Learn, XGBoost, CatBoost, LightGBM|
|Evaluation|Scikit-Learn metrics (F1, Recall, Accuracy)|
|Deployment|Streamlit|


# <a name="_toc234698628"></a>**7. Feature Engineering**
## <a name="_toc234698629"></a>**7.1 Skewness Correction**
A skewness scan across the 46 retained attributes identified 31 columns with significant skew. The Yeo-Johnson power transformation was applied to correct this, reducing the count of meaningfully skewed columns from 31 down to 17 (X11, X12, X13, X14, X15, X19, X24, X26, X37, X38, X40, X41, X42, X43, X44, X45, X51). Yeo-Johnson was chosen over the alternatives for concrete technical reasons:

- Log transformation and Box-Cox both require strictly positive values and break down on columns containing zeros — a common pattern in this dataset.
- Square-root transformation tends to work best on smaller datasets and was judged unsuitable for a dataset of this scale (458K+ rows) and variance.
- Yeo-Johnson generalizes Box-Cox to handle zero and negative values directly, making it the most robust single technique to apply uniformly across all skewed columns.

The remaining 17 columns stayed skewed even after transformation, which was noted as an accepted limitation — tree-based ensemble models (Random Forest, XGBoost, LightGBM) are comparatively robust to residual skew since they split on thresholds rather than assuming a particular distribution shape.
## <a name="_toc234698630"></a>**7.2 Feature Scaling**
Min-Max scaling was applied across the full feature set, particularly to normalize columns with wide numeric ranges, ensuring all features contribute proportionally rather than letting large-magnitude columns dominate distance- or gradient-based calculations.
## <a name="_toc234698631"></a>**7.3 Correlation Analysis & Dimensionality Reduction**
Following scaling, a correlation check revealed multicollinearity among several attributes. Principal Component Analysis (PCA) was applied to address this, configured to retain 95% of explained variance (n\_components = 0.95). This served two purposes:

- Neutralizing the correlation problem by transforming correlated features into a smaller set of uncorrelated principal components.
- Surfacing which underlying attributes contributed the most variance/impact to the model, informing feature-importance interpretation later.
## <a name="_toc234698632"></a>**7.4 Multicollinearity Check (VIF)**
Variance Inflation Factor (VIF) analysis was additionally used, independent of PCA, to directly quantify multicollinearity among the original attributes and cross-validate the PCA-based correlation findings before finalizing the modeling feature set.


# <a name="_toc234698633"></a>**8. Model Development**
## <a name="_toc234698634"></a>**8.1 Train/Test Split Strategy**
The dataset was split into training and test sets using a stratified split on the target label. Stratification was essential given the ~97.6/2.4 class imbalance: an unstratified split risks producing a training or test set with an even more skewed — or accidentally inverted — class ratio, which would degrade the reliability of both training and evaluation.
## <a name="_toc234698635"></a>**8.2 Algorithms Evaluated**
Six candidate algorithms were trained and compared: Logistic Regression, Random Forest Classifier, Decision Tree Classifier, Support Vector Machine (SVM), XGBoost Classifier, and LightGBM. All six delivered strong accuracy on the initial pass (unsurprising given the class imbalance), but recall on the minority spammer class — the metric that actually matters for this business problem — ranged only from about 51% to 71%, exposing which models were genuinely separating the classes versus which were coasting on the majority class.
## <a name="_toc234698636"></a>**8.3 Initial Ranking by Recall**

|**Rank**|**Model**|**Relative Recall Performance**|
| :- | :- | :- |
|1|LightGBM|Highest recall of the six candidates|
|2|XGBoost|Second-highest recall|
|3|Random Forest Classifier|Moderate recall|
|4|Decision Tree Classifier|Lower recall, prone to overfitting on imbalanced data|
|5|Logistic Regression|Lowest recall of the group|
## <a name="_toc234698637"></a>**8.4 Cross-Validation Strategy**
To obtain a more robust and generalizable recall estimate — and to reduce the risk that results were an artifact of one particular train/test split — Stratified K-Fold cross-validation was applied with k = 5, preserving the class ratio within every fold. This step also served as a safeguard against overfitting before committing to a final model choice.


# <a name="_toc234698638"></a>**9. Model Evaluation**
## <a name="_toc234698639"></a>**9.1 Baseline Result**
The LightGBM model was selected as the leading candidate and evaluated on the held-out test set, achieving 99% overall accuracy and 71% recall on the spammer class — already exceeding the project's success threshold of recall > 60%.
## <a name="_toc234698640"></a>**9.2 Stratified K-Fold Result**
Applying 5-fold stratified cross-validation on top of the baseline LightGBM configuration did not improve on the single-split result; the cross-validated recall came in below the original 71% figure, indicating that the initial split, while strong, may have been slightly favorable and that the model's true generalized recall sits somewhat lower.
## <a name="_toc234698641"></a>**9.3 Hyperparameter Tuning**
Hyperparameter tuning was then performed on the LightGBM model, yielding a recall of 67%. While nominally below the original 71% single-split figure, this result was considered a more trustworthy and defensible estimate of real-world performance, since it emerged from a more rigorous validation process — and it still cleared the >60% recall bar defined in the success criteria.
## <a name="_toc234698642"></a>**9.4 Final Model Summary**

|**Metric**|**Result**|
| :- | :- |
|Final Model|LightGBM (hyperparameter-tuned)|
|Accuracy|99%|
|Recall (Spammer Class)|67% (post-tuning); 71% on initial single split|
|Success Criteria|Recall > 60% — Met|
|Validation Method|Stratified 5-Fold Cross-Validation|
## <a name="_toc234698643"></a>**9.5 Interpretation**
The 99% accuracy figure, while impressive in isolation, is expected given the ~97.6% majority class share and should not be over-weighted when judging model quality — recall on the minority class remains the metric that reflects real business value here, since it measures how many actual spammers the model successfully catches.


# <a name="_toc234698644"></a>**10. Model Deployment**
The final trained LightGBM pipeline (imputation → Yeo-Johnson → Min-Max scaling → PCA → model) is serialized and wrapped in a Streamlit application to make the model usable by non-technical stakeholders such as the Trust & Safety team.
## <a name="_toc234698645"></a>**10.1 Deployment Architecture**
- Input Layer: a Streamlit form allowing either single-record manual entry or bulk CSV upload of the 46 anonymized attributes.
- Preprocessing Layer: the saved Scikit-Learn pipeline (imputer, power transformer, scaler, PCA transform) is applied identically to how it was fit during training, to avoid train/serve skew.
- Inference Layer: the tuned LightGBM model scores each record and returns both the predicted label and the predicted probability of being a spammer.
- Output Layer: results are displayed in-app as a table with a highlighted risk flag, and can be downloaded as CSV for downstream case-management workflows.
## <a name="_toc234698646"></a>**10.2 Packaging & Reproducibility**
All dependencies are pinned in requirements.txt, matching the environment used during model development, so that the Streamlit deployment reproduces the same preprocessing and inference behavior seen during evaluation.
## <a name="_toc234698647"></a>**10.3 Operational Considerations**
- A configurable probability threshold lets the Trust & Safety team tune the precision/recall trade-off without retraining the model.
- Batch scoring mode supports periodic sweeps of the full active user base rather than only real-time, one-at-a-time checks.
- Model and pipeline artifacts are versioned alongside this document (v1.0) so that predictions can always be traced back to the exact model that produced them.


# <a name="_toc234698648"></a>**11. Model Monitoring & Maintenance**
Because spam and fraud tactics evolve, a model that performs well today is not guaranteed to hold that performance indefinitely. A lightweight monitoring plan is defined to catch degradation early.
## <a name="_toc234698649"></a>**11.1 Performance Monitoring**
- Track recall and precision on a rolling window of manually-reviewed / confirmed cases to detect drift in real-world performance versus the 67–71% recall observed at evaluation time.
- Monitor the predicted-positive rate over time; a sudden spike or drop can indicate either a genuine change in spammer behavior or a data pipeline issue upstream.
## <a name="_toc234698650"></a>**11.2 Data Drift Monitoring**
- Periodically compare the distribution of incoming X1–X51 values against the training distribution (e.g., population stability index) to detect feature drift.
- Re-run the skewness and correlation checks from Section 7 on fresh data samples to confirm the existing preprocessing pipeline is still appropriate.
## <a name="_toc234698651"></a>**11.3 Retraining Cadence**
A quarterly retraining cycle is recommended as a starting cadence, with an explicit trigger for earlier retraining if monitored recall falls meaningfully below the 60% success threshold, or if data drift indicators cross a pre-agreed alert level.


# <a name="_toc234698652"></a>**12. Risks, Assumptions & Limitations**
## <a name="_toc234698653"></a>**12.1 Assumptions**
- The anonymized X1–X51 features preserve enough signal that encryption/masking does not materially harm model performance.
- The historical label distribution (~2.4% spammers) is representative of the live population the model will score.
## <a name="_toc234698654"></a>**12.2 Known Limitations**
- 17 attributes remained skewed even after Yeo-Johnson transformation, which may still affect models more sensitive to distributional assumptions than tree ensembles.
- Because features are anonymized, direct business interpretation of individual model drivers (“why was this user flagged?”) is limited to statistical importance rather than plain-language explanation.
- Recall of 67–71% means roughly 3 in 10 spammers may still go undetected by the model alone; it is intended to support, not fully replace, human review.
## <a name="_toc234698655"></a>**12.3 Risk Mitigation**
- Pairing the model with a human-in-the-loop review queue for borderline-probability cases.
- Maintaining the monitoring plan in Section 11 to catch performance decay early.


# <a name="_toc234698656"></a>**13. Conclusion & Key Learnings**
This project successfully delivered a binary classification model that meets and exceeds its defined success criteria, identifying potential spammers with 99% accuracy and 67–71% recall on a highly imbalanced dataset of 458,798 records. The end-to-end workflow — from careful missing-value and zero-variance handling, through Yeo-Johnson skew correction, Min-Max scaling, PCA-based correlation control, and stratified model validation — demonstrates a disciplined, imbalance-aware approach to a real fraud-detection problem.
## <a name="_toc234698657"></a>**13.1 Key Learnings**
- On heavily imbalanced datasets, accuracy alone is a misleading metric; recall (and F1) on the minority class is the metric that reflects actual business value.
- Stratification — both in the train/test split and in cross-validation — is essential when the target is imbalanced, to avoid misleading performance estimates.
- Tree-based ensembles (LightGBM, XGBoost) consistently outperformed Logistic Regression and simpler models on this dataset, likely because they are more robust to residual skew and complex non-linear feature interactions.
- A more rigorous validation result (post-tuning, 67% recall) can be more trustworthy than a higher but less-validated number (71% on a single split), even though it looks like a step backward on paper.


# <a name="_toc234698658"></a>**14. Future Enhancements**
- Experiment with CatBoost, which was listed among the intended frameworks but not yet benchmarked against LightGBM and XGBoost in this iteration.
- Apply class-imbalance-specific techniques (e.g., SMOTE, class-weighting, focal loss) as an alternative or complement to stratified sampling.
- Build model explainability (e.g., SHAP values on the transformed feature space) to give the Trust & Safety team more actionable insight into flagged cases despite feature anonymization.
- Extend the Streamlit application with a feedback loop so reviewer decisions (confirmed spammer / false alarm) feed back into the retraining dataset.
- Investigate GPU-accelerated training on a Linux environment to reduce iteration time for future hyperparameter searches.


# <a name="_toc234698659"></a>**15. References**
- Dataset: Kaggle — “Fiverr Predict Potential Spammers”.
- Scikit-Learn Documentation — preprocessing, PCA, and model evaluation modules.
- LightGBM, XGBoost official documentation.
- Streamlit Documentation — application deployment framework.
Page 1 of 1
