Random Forest Classification with RandomizedSearchCV
A Scikit-learn practice project using the Seaborn tips dataset.
The notebook demonstrates data preprocessing with pipelines, model
comparison, and Random Forest hyperparameter tuning using
RandomizedSearchCV.
Dataset
The project loads the dataset directly from Seaborn:
import seaborn as sns

df = sns.load_dataset("tips")
df.head()
The tips dataset contains restaurant bill and tip information. The
intended classification target in this workflow is time (Lunch or
Dinner).
Workflow
1. Load the Seaborn tips dataset.
2. Separate numerical and categorical columns.
3. Build preprocessing pipelines:
   - Numerical: median imputation and StandardScaler.
   - Categorical: most-frequent imputation and OneHotEncoder.
4. Define classification models:
   - RandomForestClassifier
   - LogisticRegression
   - DecisionTreeClassifier
5. Define a Random Forest hyperparameter search space.
6. Run RandomizedSearchCV with 10 sampled parameter combinations and
   5-fold cross-validation.
7. Inspect the best parameters and evaluate model performance.
Preprocessing
The notebook defines a numerical pipeline using:
- SimpleImputer(strategy="median")
- StandardScaler()
The categorical pipeline uses:
- SimpleImputer(strategy="most_frequent")
- OneHotEncoder()
These transformers are assembled using Scikit-learn's Pipeline and
ColumnTransformer.
Models
The notebook defines these classifiers:

  Model                 Scikit-learn estimator
  Random Forest         RandomForestClassifier()
Logistic Regression   LogisticRegression()
Decision Tree         DecisionTreeClassifier()
Random Forest Hyperparameter Search
The parameter search space provided in the notebook is:

  Parameter        Values
  max_depth      3, 5, 10, None
n_estimators   10, 20, 50, 100
criterion      gini, entropy

The search is configured with:
- n_iter=10
- cv=5
- scoring="accuracy"
- n_jobs=-1
- verbose=3
Retrieve the selected combination with:
random_search.best_params_
The exact best parameters depend on the training data and the random
search configuration.
Requirements
pip install pandas seaborn scikit-learn
Run the Notebook
1. Open the notebook in Google Colab or Jupyter Notebook.
2. Run the cells in order.
3. Check the model results and the best hyperparameters returned by
   random_search.best_params_.
Important Implementation Notes
The code snippet shared for this project is a partial workflow. Before
running it end-to-end, make sure to:
- Define x_train and y_train after selecting the target and
  splitting the data.
- Create a ColumnTransformer that applies num_pipeline to
  numerical columns and cat_pipeline to categorical columns.
- Include that preprocessor and the classifier in a single Pipeline,
  then pass the pipeline to RandomizedSearchCV.
- Define the estimator variable consistently. The snippet uses
  clasifier, which must be created before it is passed as
  estimator.
- Ensure the target column time is separated from the input
  features; do not include it in X.
Putting preprocessing inside the model pipeline ensures each
cross-validation fold learns its imputation, scaling, and encoding steps
only from that fold's training data.
Skills
Python · Pandas · Seaborn · Scikit-learn · Pipelines ·
ColumnTransformer · OneHotEncoder · Random Forest ·
RandomizedSearchCV · Cross-Validation · Classification
