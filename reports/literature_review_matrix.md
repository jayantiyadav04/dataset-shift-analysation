# Initial literature matrix and reading plan

Prepared on 2026-10-10. This is a source reading plan, not a completed critical
literature review. The assistant checked metadata and available abstracts on
primary publisher/proceedings sites. No student's full-text reading or independent
analysis is claimed. Method labels below are context; findings and limitations
remain questions to answer from full-text reading. No numerical paper results have
been imported as project results.

| ID | Starting reference and primary source | Method or context | Project relevance and next reading question | Verification and reading status |
| --- | --- | --- | --- | --- |
| R01 | Sun, Feng and Saenko (2016), [Return of Frustratingly Easy Domain Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/10306), AAAI | CORAL, second-order alignment | Compare the original transform with our declared centring and target mean restoration; derive row-vector orientation and identify assumptions. | Publisher metadata and abstract checked; full-text review pending. |
| R02 | Sugiyama, Krauledat and Müller (2007), [Covariate Shift Adaptation by Importance Weighted Cross Validation](https://www.jmlr.org/papers/v8/sugiyama07a.html), JMLR 8, 985–1005 | Covariate shift and IWCV | Explain unchanged conditional labels and source support. Distinguish IWCV from our shared source-only tuning procedure. | JMLR metadata and abstract checked; full-text review pending. |
| R03 | Huang et al., [Correcting Sample Selection Bias by Unlabeled Data](https://proceedings.neurips.cc/paper/2006/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html), NIPS 2006 proceedings | Kernel mean matching | Compare distribution matching with domain-classifier odds weighting. Determine what overlap assumptions each requires. KMM is related work only. | Proceedings metadata and abstract checked; full-text review and proceedings publication-year reconciliation pending. |
| R04 | Ben-David et al. (2010), [A theory of learning from different domains](https://link.springer.com/article/10.1007/s10994-009-5152-4), Machine Learning 79, 151–175 | Domain adaptation theory | Identify the role of discrepancy and the existence of a hypothesis suitable for both domains. Explain why the bound does not guarantee a gain. | Publisher metadata and abstract checked; issue year 2010, online publication 2009; full-text review pending. |
| R05 | Blitzer, Dredze and Pereira (2007), [Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification](https://aclanthology.org/P07-1056/), ACL, 440–447 | Cross-category sentiment benchmark | Read the benchmark and structural correspondence learning setup; distinguish that method from this study's TF-IDF/SVD and two adaptations. | ACL metadata checked; full-text and corpus-version review pending. |
| R06 | Fernando, Habrard, Sebban and Tuytelaars (2013), [Unsupervised Visual Domain Adaptation Using Subspace Alignment](https://openaccess.thecvf.com/content_iccv_2013/html/Fernando_Unsupervised_Visual_Domain_2013_ICCV_paper.html), ICCV, 2960–2967 | Subspace alignment | Compare subspaces with covariance alignment. Which feature assumptions differ from this study's shared dense representation? | Direct page opening failed initially; publisher-indexed metadata and abstract checked through search; full-text review pending. |
| R07 | Pan and Yang (2010), [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526), IEEE TKDE 22(10) | Transfer-learning taxonomy | Define domain, task, transfer learning and unsupervised adaptation in the terminology relevant to the frozen project. | Direct page content was restricted; IEEE-indexed metadata checked and an [author-hosted copy](https://home.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf) identified; full-text review pending. |
| R08 | Vergara et al. (2012), [Chemical gas sensor drift compensation using classifier ensembles](https://www.sciencedirect.com/science/article/pii/S0925400512002018), Sensors and Actuators B 166–167, 320–329, DOI 10.1016/j.snb.2012.01.074 | Gas drift dataset context | Read the collection process and temporal dependence. Explain why this study uses Batch 1 to later batches and does not reproduce the paper's ensemble. | DOI opening failed initially; publisher-indexed metadata and abstract checked through search; full-text and raw archive review pending. |
| R09 | Anguita, Ghio, Oneto, Parra and Reyes-Ortiz (2013), [A Public Domain Dataset for Human Activity Recognition Using Smartphones](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf), ESANN | HAR dataset construction | Read participant partitions and overlapping windows. Explain how participant grouping protects validation and evaluation. | Proceedings PDF title/authors checked; full-text and official archive review pending. |
| R10 | Pedregosa et al. (2011), [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html), JMLR 12, 2825–2830 | Implementation library | Cite the library and connect estimator APIs to the actual locked environment; distinguish software attribution from method attribution. | JMLR source checked; full-text review pending. |

## Reading order and evidence to add

Start with R01 and R02 to explain the two adaptation mechanisms and their limits.
Next read R05, R08 and R09 for dataset provenance and split design. R04 and R07
support terminology and assumptions; R03 and R06 supply related alternatives;
R10 supports software attribution.

For each full-text reading, record the actual date, exact setup, main finding with
its page/table reference, limitation, and how it informs this project. Preserve
negative or conditional findings. Add only verified relevant papers as needed for
the final review; the guide's recommended final reference total has not been met
by this starting matrix alone.
