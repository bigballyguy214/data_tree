"""Run: python analysis.py. Outputs reproduce the complete analysis."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
FEATURES = ['lead_time_days', 'prior_no_shows', 'age', 'distance_miles',
            'sms_reminder', 'new_patient', 'appointment_hour']
SEED = 42
ANNUAL_VOLUME = 10000

def appointment_decision(appointment, model):
    """Use the fitted tree's majority-class prediction, as assigned."""
    row = pd.DataFrame([appointment], columns=FEATURES)
    return 'overbook' if model.predict(row)[0] == 1 else 'hold'

def main():
    df = pd.read_csv(ROOT / 'clinic_appointments.csv')
    assert df.shape == (400, 8)
    assert list(df.columns) == FEATURES + ['no_show']
    assert not df.isna().any().any()
    assert set(df.no_show) == {'no-show', 'checked-in'}
    y = df.no_show.eq('no-show').astype(int)
    assert y.sum() == 91
    # Split before fitting. Hyperparameters fixed in advance; no test-set tuning.
    train, test = train_test_split(np.arange(len(df)), test_size=.30,
                                  stratify=y, random_state=SEED)
    X = df[FEATURES]
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15,
                                   random_state=SEED)
    model.fit(X.iloc[train], y.iloc[train])
    metrics = []
    for name, pred in [('baseline', np.zeros(len(test), dtype=int)),
                       ('v1', model.predict(X.iloc[test]))]:
        tn, fp, fn, tp = confusion_matrix(y.iloc[test], pred, labels=[0,1]).ravel()
        value = -60*int(fp) -120*int(fn)
        metrics.append(dict(model=name, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
            accuracy=accuracy_score(y.iloc[test], pred),
            precision=precision_score(y.iloc[test], pred, zero_division=0),
            recall=recall_score(y.iloc[test], pred, zero_division=0),
            total_value=value, value_per_appointment=value/len(test),
            annual_value_at_10000=value/len(test)*ANNUAL_VOLUME))
    upcoming = {
        'D': dict(zip(FEATURES,[21,1,27,9.5,0,1,15])),
        'E': dict(zip(FEATURES,[3,0,61,2.0,1,0,9]))}
    decisions = {}
    for name, row in upcoming.items():
        frame = pd.DataFrame([row], columns=FEATURES)
        leaf = int(model.apply(frame)[0])
        path = model.decision_path(frame).indices
        steps=[]
        for node in path:
            feature=model.tree_.feature[node]
            if feature >= 0:
                field=FEATURES[feature]; threshold=float(model.tree_.threshold[node])
                steps.append(f'{field} {"<=" if row[field] <= threshold else ">"} {threshold:g}')
        counts = y.iloc[train][model.apply(X.iloc[train]) == leaf].value_counts()
        decisions[name] = dict(appointment=row, leaf=leaf, path=steps,
            decision=appointment_decision(row,model),
            training_leaf_shows=int(counts.get(0,0)), training_leaf_no_shows=int(counts.get(1,0)))
    result = dict(source='User-supplied clinic_appointments.csv; original collection source unspecified',
        rows=len(df), no_shows=int(y.sum()), seed=SEED,
        train_rows=len(train), train_no_shows=int(y.iloc[train].sum()),
        test_rows=len(test), test_no_shows=int(y.iloc[test].sum()),
        features=FEATURES, max_depth=3, min_samples_leaf=15,
        actual_depth=model.get_depth(), training_accuracy=model.score(X.iloc[train],y.iloc[train]),
        metrics=metrics, upcoming=decisions, annual_volume_assumption=ANNUAL_VOLUME,
        savings_per_appointment=metrics[1]['value_per_appointment']-metrics[0]['value_per_appointment'],
        python=sys.version.split()[0], sklearn=sklearn.__version__, pandas=pd.__version__)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    tree = export_text(model,feature_names=FEATURES)
    (OUT/'tree_rules.txt').write_text(tree)
    pd.DataFrame(metrics).to_csv(OUT/'test_metrics.csv', index=False)
    audit=df.iloc[test].copy(); audit.insert(0,'csv_row_number',test+2)
    audit['v1_prediction']=np.where(model.predict(X.iloc[test]),'no-show','checked-in')
    audit['v1_value']=np.where(model.predict(X.iloc[test]),np.where(y.iloc[test],0,-60),np.where(y.iloc[test],-120,0))
    audit.to_csv(OUT/'test_predictions.csv',index=False)
    pd.DataFrame({'csv_row_number':np.arange(len(df))+2,'partition':np.where(np.isin(np.arange(len(df)),train),'train','test')}).to_csv(OUT/'split.csv',index=False)
    fig,ax=plt.subplots(figsize=(16,7))
    plot_tree(model,feature_names=FEATURES,class_names=['show','no-show'],filled=True,
              rounded=True,node_ids=True,impurity=False,ax=ax,fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/'tree.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    vals=[r['value_per_appointment'] for r in metrics]
    ax.bar(['Hold every slot','Tree policy'],vals,color=['#999999','#397ba8'])
    for i,v in enumerate(vals): ax.text(i,v-1,f'${v:.2f}',ha='center',va='top')
    ax.axhline(0,color='black',linewidth=.8); ax.set_ylim(min(vals)-7,1)
    ax.set_ylabel('Value per test appointment ($)'); fig.tight_layout()
    fig.savefig(OUT/'value_comparison.png',dpi=200); plt.close(fig)
    print(json.dumps(result,indent=2)); print(tree)

if __name__ == '__main__':
    main()
