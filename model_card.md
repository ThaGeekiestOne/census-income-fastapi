# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a random forest classifier built with scikit-learn (version 1.7.2) that predicts whether a person earns more than $50,000 per year. It uses 100 trees with a minimum of three samples per leaf and a fixed random seed of 42 so that training is reproducible. The categorical inputs are one-hot encoded and the salary label is binarized, and the fitted encoder and label binarizer are saved next to the model in the `model/` directory. The model is served through a FastAPI application and was developed as part of the Udacity Machine Learning DevOps Engineer course project.

## Intended Use

The model is intended for learning how to train, test, version, and deploy a machine learning model with continuous integration and continuous deployment. Its intended users are students and reviewers who want to see an end-to-end example of a classification model behind a REST API. It is not intended for real decisions about people, such as hiring, lending, or pricing, because it was trained on historical data and has not been validated for those purposes.

## Training Data

The model was trained on the Census Income (Adult) dataset from the UCI Machine Learning Repository, which was extracted from a 1994 U.S. Census Bureau database. The file contains 32,561 rows and 15 columns: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, and the salary label. About 24 percent of the rows are labeled `>50K` (7,841 rows) and about 76 percent are labeled `<=50K` (24,720 rows). The only cleaning step was removing the stray spaces after commas in the raw file, so values such as `?` in workclass, occupation, and native-country were kept as their own categories. The data was split 80/20 into training and test sets with a stratified split on the salary label, which leaves 26,048 rows for training.

## Evaluation Data

The model was evaluated on the 20 percent hold-out test set, which contains 6,513 rows with the same proportion of salary labels as the full dataset. The test rows were not used for fitting the model or the encoder. They were also used to compute the slice metrics described below.

## Metrics

The model is evaluated with precision, recall, and F1 score for the positive class (`>50K`). Precision is the share of people predicted to earn more than $50K who actually do, recall is the share of people who actually earn more than $50K that the model finds, and F1 is the harmonic mean of the two. On the test set the model achieves a precision of 0.7841, a recall of 0.6276, and an F1 score of 0.6971.

Performance was also computed on slices of the test data for every value of every categorical feature, and the full results are in `slice_output.txt`. A few examples illustrate how performance varies:

- By sex, precision is 0.8529 for women and 0.7733 for men, while recall is 0.5918 for women and 0.6342 for men.
- By race, the F1 score is 0.7020 for White, 0.6452 for Black, 0.6602 for Asian-Pac-Islander, and 0.4615 for Amer-Indian-Eskimo individuals.
- By education, the F1 score is high for Masters (0.8723) and Bachelors (0.7922) but low for HS-grad (0.4325) and for 11th grade (0.4286), mostly because recall is low for those groups.

Several slices contain very few rows, such as Other for race (45 rows) or Never-worked for workclass (1 row), so their scores are not reliable.

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex, and native country, and the model uses them as input features. The slice results show that performance is not uniform across groups. For example, recall is lower for Black individuals (0.5376) than for White individuals (0.6333), so the model is more likely to miss high earners in some groups than in others. The data also reflects the structure of the labor market in 1994, so it can encode historical inequalities, and a model trained on it can reproduce them. Because of this, the model should not be used to make or support decisions that affect people's opportunities.

## Caveats and Recommendations

The data is from 1994, so the income threshold, occupations, and demographics do not match today's population, and the model should not be expected to generalize to current data. The classes are imbalanced, and the model favors precision over recall, which means it misses roughly a third of the people who earn more than $50K. Slices with few examples produce noisy metrics, so conclusions about small groups should be drawn carefully. Anyone who wants to use a model like this more seriously should retrain it on current, representative data, tune the decision threshold for their use case, evaluate fairness across groups before deployment, and monitor performance after deployment. The model was trained and tested with scikit-learn 1.7.2, and loading the saved artifacts with a different version may fail or give different results.
