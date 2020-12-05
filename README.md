### Anomaly-detection using Multivariate Gaussian distribution

- Data acquired from Kaggle:https://www.kaggle.com/mlg-ulb/creditcardfraud
  it contains 2-day transactions made by credit cards in September 2013 by European cardholders.
- Data contain 30 features and binary class notification; y0=normal transaction, y1=Fraud.
- Conducted Anomaly detection using Multivariate Gaussian distribution method.
- Model evaluation was conducted using testset and acquired AUC 0.81

Q.Anomaly detection vs. supervised classification, which one should I apply? <br>
Anomaly detection can be suitable when there's a very small number of positive examples(anomalies)
since it's hard for a model to learn the positive examples from the sample given. 
Also, anomalies might look nothing like any other in the training dataset. 
In this dataset, y1 takes up 0.17%, highly imbalanced dataset.


Multivariate Gaussian distribution
$\mu\in\mathbb{R}$
<br>
$\Sigma \in \mathbb{R}^{nxn} $ (covariance matrix)
<br>
$p(x;\mu,{\boldsymbol\Sigma})=\frac{1}{\sqrt{(2\pi)^n|\boldsymbol\Sigma|}}
\exp\left(-\frac{1}{2}({x}-{m})^T{\boldsymbol\Sigma}^{-1}({x}-{m})
\right)$

procedures:
1. Choose features that might be indicative of anomalis examples (already done above feature importance section)
2. Fit paraters; $\mu$, $\Sigma$ based on the training set
3. On cross-validation set, predict <br>
   y=1, if p(x) $< \epsilon$ <br>
   y=0, if p(x) $\geq \epsilon$ <br>
   Select best epsilon using evaluation metrics.<br>
4. On testset, Evaluate the model using evaluation metrics.

Pro and Cons of Multivariate Gaussian distribution:
it automatically captures correlations between features.
However, it computationaly more expensive and the number of samples should be enough.


references
- Andrew Ng's Anomaly detection ML lectures (https://www.youtube.com/watch?v=086OcT-5DYI&list=PLwgXNx7TiGV6UH3aEzmdZwzFRwvEnRb0N&ab_channel=ArtificialIntelligence-AllinOne)
- Sachin Shelar's Kaggle notebook (https://www.kaggle.com/shelars1985/anomaly-detection-using-gaussian-distribution)

