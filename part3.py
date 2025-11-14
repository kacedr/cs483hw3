from part1 import load_and_clean, prepare
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,brier_score_loss

from fairlearn.metrics import demographic_parity_difference,equal_opportunity_difference
from fairlearn.reductions import ExponentiatedGradient,DemographicParity
from sklearn.utils import resample

df=load_and_clean("data")
df=prepare(df)

target_col="y"
sens_col="sex"

sensitive=df[sens_col]
X=df.drop(columns=["income",target_col,sens_col,"sex_male"],errors="ignore")
y=df[target_col]

X_train,X_test,y_train,y_test,s_train,s_test=train_test_split(X,y,sensitive,test_size=0.3,random_state=42,stratify=y)

numeric_cols=X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols=X.select_dtypes(include=["object"]).columns.tolist()

preprocess=ColumnTransformer(transformers=[("num",StandardScaler(with_mean=False),numeric_cols),
("cat",OneHotEncoder(handle_unknown="ignore"),categorical_cols)],remainder="drop")

def eval_model(name,y_true,y_pred,y_prob,s):
        acc=accuracy_score(y_true,y_pred)


        prec=precision_score(y_true,y_pred,zero_division=0)
        rec=recall_score(y_true,y_pred,zero_division=0)
        f1=f1_score(y_true,y_pred,zero_division=0)

        b=brier_score_loss(y_true,y_prob)
        dp=demographic_parity_difference(y_true,y_pred,sensitive_features=s)
        eo=equal_opportunity_difference(y_true,y_pred,sensitive_features=s)

        vals={"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"brier":b,"dp_diff":dp,"eo_diff":eo}
        return pd.Series(vals,name=name)

baseline_clf=Pipeline(steps=[("prep",preprocess),("lr",LogisticRegression(max_iter=1000))])
baseline_clf.fit(X_train,y_train)

y_pred_base=baseline_clf.predict(X_test)
y_prob_base=baseline_clf.predict_proba(X_test)[:,1]

baseline_row=eval_model("baseline",y_test,y_pred_base,y_prob_base,s_test)
print("Baseline metrics:")
print(baseline_row.to_string())
print("")



train_df=X_train.copy()
train_df[target_col]=y_train.values
train_df[sens_col]=s_train.values

groups=train_df.groupby([sens_col,target_col])
min_size=groups.size().min()

resampled_parts=[]

for key,grp in groups:
    r=resample(grp,replace=(len(grp)<min_size),n_samples=min_size,random_state=42)
    resampled_parts.append(r)



train_resampled=pd.concat(resampled_parts,axis=0).reset_index(drop=True)



X_train_res=train_resampled.drop(columns=[target_col,sens_col])

y_train_res=train_resampled[target_col]
s_train_res=train_resampled[sens_col]

resampled_clf=Pipeline(steps=[("prep",preprocess),("lr",LogisticRegression(max_iter=1000))])


resampled_clf.fit(X_train_res,y_train_res)

y_pred_res=resampled_clf.predict(X_test)

y_prob_res=resampled_clf.predict_proba(X_test)[:,1]

resampled_row=eval_model("preprocess_resample",y_test,y_pred_res,y_prob_res,s_test)
print("Pre-processing (resampling) metrics:")
print(resampled_row.to_string())
print("")




X_train_proc=preprocess.fit_transform(X_train).toarray()
X_test_proc=preprocess.transform(X_test).toarray()


base_logreg=LogisticRegression(max_iter=1000)
dp_constraint=DemographicParity()


mitigator=ExponentiatedGradient(estimator=base_logreg,constraints=dp_constraint,eps=0.01)
mitigator.fit(X_train_proc,y_train,sensitive_features=s_train)



y_pred_fair=mitigator.predict(X_test_proc)
probs_fair=mitigator._pmf_predict(X_test_proc)
sums=probs_fair.sum(axis=1,keepdims=True)
probs_fair=probs_fair/sums

y_prob_fair=probs_fair[:,1]
y_prob_fair=np.clip(y_prob_fair,0,1)

expgrad_row=eval_model("inproc_expgrad_DP",y_test,y_pred_fair,y_prob_fair,s_test)

print("In-processing (ExponentiatedGradient) metrics:")
print(expgrad_row.to_string())
print("")



results_df=pd.DataFrame([baseline_row,resampled_row,expgrad_row])
print("=== Summary: baseline vs mitigation ===")
print(results_df.round(3).to_string())
