# HuMob (BTP-2)

## Statistical baselines

| Metric                    | City A   | City B   | City C   | City D   | Average  |
|---------------------------|----------|----------|----------|----------|----------|
| Global Mean               | 0.00029  | 0.00025  | 0.00107  | 0.00001  | 0.00040  |
| Global Mode               | 0.00126  | 0.00419  | 0.00493  | 0.00191  | 0.00308  |
| Per-User Mean             | 0.01651  | 0.01778  | 0.01588  | 0.02026  | 0.01761  |
| Per-User Mode             | 0.07918  | 0.08420  | 0.08338  | 0.08866  | 0.08385  |
| Unigram Model             | 0.03585  | 0.04105  | 0.03726  | 0.04309  | 0.03932  |
| Bigram Model              | 0.05243  | 0.05974  | 0.05380  | 0.06066  | 0.05666  |
| Bigram Model (top_p=0.7)  | 0.07819  | 0.08984  | 0.08002  | 0.09044  | 0.08463  |

## TTKNN models

| Metric                    | City A   | City B   | City C   | City D   | Average  |
|---------------------------|----------|----------|----------|----------|----------|
| TT-KNN                    | 0.08741  | 0.09649  | 0.08558  | 0.09782  | 0.09183  |
| Freq TT-KNN               | 0.09603  | 0.10185  | 0.09032  | 0.10250  | 0.09768  |
| Cluster TT-KNN(kmeans)    | 0.10480  | 0.10626  | 0.09532  | 0.10638  | 0.10319  |
| Cluster TT-KNN(Birch)     | 0.10532  | 0.10679  | 0.09621  | 0.10660  | 0.10373  |
| Cluster TT-KNN(HDBSCAN)   | 0.10572  | 0.10724  | 0.09708  | 0.10748  | 0.10438  |
<!-- | SVR(LightGBM + XGBoost)   | -        | 0.1390   | 0.1493   | 0.1536   | 0.1473   | -->

## presention

- mid presentation : https://www.canva.com/design/DAGysLr3Pd8/zHUAVrSGkmUtzH9XXhz-TQ/edit?utm_content=DAGysLr3Pd8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Important Dates

- Data open: April 30 (Wed), 2025
- Midterm evaluation submission deadline (optional): 11:59 pm AoE, July 30 (Wed), 2025
- Midterm evaluation result announcement: August 20 (Wed), 2025
- Submission deadline: 11:59 pm AoE, September 10 (Wed), 2025
- Result announcement & Invitation of papers for submissions: October 1 (Wed), 2025
- Submission deadline for invited papers: 11:59 pm AoE, October 15 (Wed), 2025
- GISCUP presentations: November 5 (Wed), 2025
