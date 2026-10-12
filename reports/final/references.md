# Verified references

Checked against the primary publisher, proceedings, author or official software
source on 11 October 2026. Method and dataset claims below are attributed to their
original sources. Project scores come from the saved local study, not these papers.

R01. Sun, B., Feng, J. and Saenko, K. (2016). Return of Frustratingly Easy Domain
Adaptation. Proceedings of AAAI, 30(1). DOI: 10.1609/aaai.v30i1.10306.
[Publisher and paper](https://ojs.aaai.org/index.php/AAAI/article/view/10306).

R02. Sugiyama, M., Krauledat, M. and Müller, K.-R. (2007). Covariate Shift
Adaptation by Importance Weighted Cross Validation. JMLR, 8, 985-1005.
[Paper](https://www.jmlr.org/papers/v8/sugiyama07a.html).

R03. Huang, J., Smola, A. J., Gretton, A., Borgwardt, K. M. and Schölkopf, B.
(2007). Correcting Sample Selection Bias by Unlabeled Data. Advances in Neural
Information Processing Systems 19, proceedings of NIPS 2006.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf).
Author order follows the paper PDF; the online proceedings metadata orders authors differently.

R04. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F. and
Vaughan, J. W. (2010). A theory of learning from different domains. Machine
Learning, 79, 151-175. DOI: 10.1007/s10994-009-5152-4.
[Article](https://link.springer.com/article/10.1007/s10994-009-5152-4).
Online publication was in 2009; the cited journal volume is 2010.

R05. Blitzer, J., Dredze, M. and Pereira, F. (2007). Biographies, Bollywood,
Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification.
Proceedings of ACL, 440-447.
[Paper](https://aclanthology.org/P07-1056/).

R06. Fernando, B., Habrard, A., Sebban, M. and Tuytelaars, T. (2013).
Unsupervised Visual Domain Adaptation Using Subspace Alignment. ICCV, 2960-2967.
[Proceedings](https://openaccess.thecvf.com/content_iccv_2013/html/Fernando_Unsupervised_Visual_Domain_2013_ICCV_paper.html).

R07. Pan, S. J. and Yang, Q. (2010). A Survey on Transfer Learning. IEEE
Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.
DOI: 10.1109/TKDE.2009.191.
[Publisher](https://ieeexplore.ieee.org/document/5288526);
[author manuscript](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf).

R08. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L. and
Huerta, R. (2012). Chemical gas sensor drift compensation using classifier
ensembles. Sensors and Actuators B: Chemical, 166-167, 320-329.
DOI: 10.1016/j.snb.2012.01.074.
[Publisher](https://www.sciencedirect.com/science/article/pii/S0925400512002018).

R09. Anguita, D., Ghio, A., Oneto, L., Parra, X. and Reyes-Ortiz, J. L. (2013).
A Public Domain Dataset for Human Activity Recognition Using Smartphones.
ESANN, 437-442.
[Proceedings paper](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf).

R10. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
JMLR, 12, 2825-2830.
[Paper](https://www.jmlr.org/papers/v12/pedregosa11a.html).

R11. Bickel, S., Brückner, M. and Scheffer, T. (2009). Discriminative Learning
Under Covariate Shift. JMLR, 10, 2137-2155.
[Paper](https://www.jmlr.org/papers/v10/bickel09a.html).

D01. Vergara, A. (2012). Gas Sensor Array Drift Dataset. UCI Machine Learning
Repository, dataset 224. DOI: 10.24432/C5RP6W.
[Dataset](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset).

D02. Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. and Parra, X. (2013).
Human Activity Recognition Using Smartphones. UCI Machine Learning Repository,
dataset 240. DOI: 10.24432/C54S4K.
[Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

D03. Dredze, M. Multi-Domain Sentiment Dataset, original four-domain release.
Author-hosted distribution, page last updated 2 August 2007; cite R05.
[Dataset and separate books file](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html).

S01. Scikit-learn developers. LogisticRegression, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

S02. Scikit-learn developers. LinearSVC, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

S03. Scikit-learn developers. RandomForestClassifier, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

S04. Scikit-learn developers. StratifiedGroupKFold, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

S05. Scikit-learn developers. TfidfVectorizer, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

S06. Scikit-learn developers. TruncatedSVD, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

## Verification scope

R01-R05, R07, R09-R11 were checked using primary full-paper text, including the
method or dataset sections used here. R06 was checked using the indexed CVF
proceedings abstract and metadata; direct HTML/PDF access returned HTTP 403.
R08 was checked using the indexed publisher abstract, introduction and method
preview; direct publisher access also returned HTTP 403. No unseen experimental
detail from either restricted paper is claimed. IEEE's page required JavaScript;
the author-hosted R07 manuscript supplied the text. The DOI, volume and page
metadata identify the final publication separately from that manuscript.

D01-D03 and S01-S06 were opened directly. The online scikit-learn documentation
identified itself as 1.9.1, matching the installed scientific environment.
The reading matrix records limitations and relevance; it does not claim that
the student independently read or verified every source.
