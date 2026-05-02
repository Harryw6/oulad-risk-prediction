# 论文图表整理说明

本文件整理当前实验已经生成、可用于论文写作的表格和图形。所有数值均来自已执行代码生成的 CSV 或图像文件，不包含额外推断或未执行实验结果。

## 表 1 数据集统计

建议表题：**表 1 OULAD 数据集统计与标签分布**

数据来源：outputs/tables/dataset_summary.csv

建议内容：

| 指标 | 数值 |
| --- | ---: |
| 学生-课程-开课学期实例数 | 32,593 |
| 唯一学生数 | 28,785 |
| 课程开课学期数 | 22 |
| VLE 交互记录数 | 10,655,280 |
| 学生评估记录数 | 173,912 |
| 非风险实例数 | 15,385 |
| 风险实例数 | 17,208 |
| 风险实例比例 | 0.527966 |

正文引用建议：该表用于说明实验样本规模、标签定义后的类别分布以及数据源规模。

## 表 2 第 56 天模型比较结果

建议表题：**表 2 第 56 天预测窗口下不同模型的性能比较**

数据来源：outputs/tables/model_comparison.csv

建议内容保留主验证协议 `stratified_student_group_train_test` 下的模型结果，并重点列出 Accuracy、Precision、Recall、F1、ROC-AUC、PR-AUC 和 Balanced Accuracy。可以按 PR-AUC 降序排列。当前结果中 CatBoost 的 PR-AUC 最高，为 0.9186486964962812；XGBoost 的 PR-AUC 为 0.9185809528222727，二者非常接近。

正文引用建议：该表用于说明 CatBoost 在主模型比较中取得最高 PR-AUC，而 XGBoost 与其结果接近，并被用于后续窗口、消融和 SHAP 分析。

## 表 3 跨课程开课学期验证结果

建议表题：**表 3 课程开课学期分组验证性能**

数据来源：outputs/tables/model_comparison.csv

建议内容：

| 验证协议 | 模型 | 折数 | PR-AUC | ROC-AUC | F1 | Balanced Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GroupKFold by course presentation | XGBoost | 5 | 0.9106072369677266 | 0.8803302147302228 | 0.7969606110111422 | 0.7954748459839097 |
| Leave-one-course-presentation-out | XGBoost | 22 | 0.868957712485784 | 0.8663078252476215 | 0.7514300218132678 | 0.7700311214180374 |

正文引用建议：该表用于讨论课程开课学期迁移场景下的性能下降，说明跨课程泛化比主学生分组训练-测试更困难。

## 表 4 多窗口早期预测结果

建议表题：**表 4 多预测窗口下 XGBoost 的早期预警性能**

数据来源：outputs/tables/time_window_results.csv

建议内容：

| 预测窗口 | PR-AUC | ROC-AUC | F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第 7 天 | 0.8041911779170687 | 0.7789676154377372 | 0.7145421903052065 | 0.7081198115311498 | 0.7073170731707317 |
| 第 14 天 | 0.8254951083251584 | 0.7927083417917458 | 0.7155824508320726 | 0.7130660235818335 | 0.7116122104617273 |
| 第 28 天 | 0.8832185448773401 | 0.8503195721966335 | 0.7609740964789825 | 0.7666357694631138 | 0.7636140512348519 |
| 第 56 天 | 0.9185809528222727 | 0.8918490866897415 | 0.8092165898617512 | 0.812103143092544 | 0.8094799815922687 |
| 完整课程 | 0.9894895678678277 | 0.985792227652182 | 0.9474157303370786 | 0.9477887617016432 | 0.9461573861021629 |

正文引用建议：该表用于论证随着观测窗口延长，风险预测性能逐步提高；完整课程窗口仅作为上界比较。

## 表 5 特征消融结果

建议表题：**表 5 不同特征组在多预测窗口下的 PR-AUC**

数据来源：outputs/tables/feature_ablation_results.csv

建议内容可优先呈现 PR-AUC，以突出不同信息源的贡献：

| 特征组 | 第 7 天 | 第 14 天 | 第 28 天 | 第 56 天 | 完整课程 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Demographic only | 0.6711683980597183 | 0.6714085190813539 | 0.6718281289333156 | 0.6707936428224462 | 0.6721860046123783 |
| VLE behavior only | 0.7579831232784434 | 0.7906172551642031 | 0.8286905118028831 | 0.8757945592173886 | 0.9775490616226298 |
| Assessment only | 0.5279950912716674 | 0.5465226996626236 | 0.8361077812002313 | 0.9019258000234883 | 0.9873322850982235 |
| Demographic + VLE | 0.8037004457739649 | 0.824310163656357 | 0.8505472661451849 | 0.8882076116219266 | 0.9779468262275905 |
| Demographic + VLE + Assessment | 0.8041911779170687 | 0.8254951083251584 | 0.8832185448773401 | 0.9185809528222727 | 0.9894895678678277 |

正文引用建议：该表用于说明早期窗口中 VLE 行为较为关键，而评估特征在第 28 天和第 56 天后显著增强；组合特征组整体表现最好。

## 表 6 SHAP Top 特征

建议表题：**表 6 XGBoost 模型的 Top 20 SHAP 特征重要性**

数据来源：outputs/tables/shap_top_features.csv

建议正文重点报告前 10 个特征：

| 排名 | 特征 | Mean absolute SHAP |
| ---: | --- | ---: |
| 1 | `vle_days_since_last_activity` | 0.7940888 |
| 2 | `assessment_missing_due_count` | 0.7315784 |
| 3 | `assessment_mean_score` | 0.27383345 |
| 4 | `assessment_weighted_mean_score` | 0.19731991 |
| 5 | `assessment_completed_weight` | 0.19602647 |
| 6 | `assessment_due_weight` | 0.1803406 |
| 7 | `assessment_average_delay` | 0.14879794 |
| 8 | `highest_education_Lower Than A Level` | 0.14576882 |
| 9 | `vle_clicks_activity_page` | 0.1051074 |
| 10 | `vle_active_day_ratio` | 0.100095406 |

正文引用建议：该表用于解释模型主要依赖平台参与间隔、缺失评估、评估得分和评估完成情况等信息。

## 表 7 多窗口 bootstrap 置信区间

建议表题：**表 7 多预测窗口下 XGBoost 性能的 bootstrap 95% 置信区间**

数据来源：outputs/tables/bootstrap_ci_time_window_results.csv

建议内容：报告每个预测窗口的 PR-AUC、PR-AUC 95% CI、F1 和 F1 95% CI。该表可用于增强结果的统计稳健性说明。

## 表 8 逐课程开课学期留一验证明细

建议表题：**表 8 Leave-one-course-presentation-out 逐课程开课学期性能**

数据来源：outputs/tables/per_course_presentation_results.csv

建议内容：报告每个 held-out course presentation 的样本数、风险比例、PR-AUC、ROC-AUC、F1 和 Balanced Accuracy。该表可放入附录或正文压缩展示，用于说明跨课程泛化存在异质性。

## 表 9 退课样本敏感性分析

建议表题：**表 9 排除预测日前已退课样本后的敏感性分析**

数据来源：outputs/tables/withdrawal_sensitivity_results.csv

建议内容：报告每个预测窗口过滤前样本量、过滤后样本量、排除样本数、过滤后风险比例、PR-AUC 和 F1。该表用于讨论 Withdrawn 标签对早期预警任务的影响。

## 表 10 校准与阈值分析

建议表题：**表 10 第 56 天 XGBoost 的校准与阈值分析**

数据来源：

- outputs/tables/calibration_summary_day56.csv
- outputs/tables/calibration_bins_day56.csv
- outputs/tables/threshold_analysis_day56.csv

建议内容：正文中报告 Brier score、10-bin ECE、平均预测风险和实际风险比例；阈值表可报告若干代表阈值下的 precision、recall 和 F1。

## 表 11 多随机种子模型比较

建议表题：**表 11 第 56 天多随机种子重复模型比较**

数据来源：

- outputs/tables/repeated_seed_model_results.csv
- outputs/tables/repeated_seed_model_summary.csv

建议内容：报告每个模型在 10 个学生分组随机划分下的 PR-AUC 均值、标准差、F1 均值、F1 标准差、ROC-AUC 均值和平衡准确率均值。当前结果中 XGBoost 的平均 PR-AUC 为 0.9181163962207144，LightGBM 为 0.9176360017687116，CatBoost 为 0.917397192442683。

## 表 12 多窗口重复实验与显著性检验

建议表题：**表 12 多预测窗口重复实验与配对显著性检验**

数据来源：

- outputs/tables/repeated_seed_window_results.csv
- outputs/tables/repeated_seed_window_summary.csv
- outputs/tables/significance_tests.csv

建议内容：报告 XGBoost 在各预测窗口下的 PR-AUC/F1 均值和标准差，并单独报告 day_56 相对 day_7、day_14、day_28 的 Wilcoxon signed-rank 检验结果。当前结果显示 day_56 相对更早窗口的 PR-AUC、F1、ROC-AUC 和 balanced accuracy 提升在 Holm 校正后均达到 p < 0.05。

## 图 1 标签分布

建议图题：**图 1 风险与非风险学生实例的标签分布**

图像路径：outputs/figures/label_distribution.png

用途：展示 `at_risk = 0` 与 `at_risk = 1` 的样本数量，为类别不平衡处理和指标选择提供背景。

## 图 2 多窗口预测性能

建议图题：**图 2 不同预测窗口下模型性能变化**

图像路径：outputs/figures/time_window_performance.png

用途：展示第 7 天、第 14 天、第 28 天、第 56 天和完整课程窗口下的指标变化趋势。

## 图 3 ROC 曲线

建议图题：**图 3 第 56 天模型比较的 ROC 曲线**

图像路径：outputs/figures/roc_curves.png

用途：展示不同模型在第 56 天主比较实验中的 ROC 曲线。

## 图 4 PR 曲线

建议图题：**图 4 第 56 天模型比较的 Precision-Recall 曲线**

图像路径：outputs/figures/pr_curves.png

用途：在类别不平衡背景下比较不同模型的 precision-recall 表现。

## 图 5 最佳模型混淆矩阵

建议图题：**图 5 第 56 天最佳主比较模型的混淆矩阵**

图像路径：outputs/figures/confusion_matrix_best_model.png

用途：展示主模型比较中最佳模型的错误类型，包括假阳性和假阴性数量。

## 图 6 SHAP 全局解释图

建议图题：**图 6 XGBoost 模型的 SHAP 全局特征贡献分布**

图像路径：outputs/figures/shap_summary.png

用途：展示全局范围内主要特征对预测输出的贡献方向和大小。

## 图 7 SHAP 依赖图

建议图题：**图 7 主要特征的 SHAP 依赖关系**

图像路径：

- outputs/figures/shap_dependence_vle_days_since_last_activity.png
- outputs/figures/shap_dependence_assessment_missing_due_count.png
- outputs/figures/shap_dependence_assessment_mean_score.png
- outputs/figures/shap_dependence_assessment_weighted_mean_score.png
- outputs/figures/shap_dependence_assessment_completed_weight.png

用途：辅助解释主要特征取值与模型输出贡献之间的关系。

## 图 8 校准曲线

建议图题：**图 8 第 56 天 XGBoost 风险概率校准曲线**

图像路径：outputs/figures/calibration_curve_day56.png

用途：展示模型预测风险概率与实际风险比例之间的一致性。

## 图 9 阈值权衡曲线

建议图题：**图 9 第 56 天 XGBoost 不同决策阈值下的性能权衡**

图像路径：outputs/figures/threshold_tradeoff_day56.png

用途：展示不同预警阈值下 precision、recall、F1 和 balanced accuracy 的变化，为实际干预策略提供参考。

## 图 10 多随机种子模型 PR-AUC 分布

建议图题：**图 10 第 56 天不同模型在多随机种子划分下的 PR-AUC 分布**

图像路径：outputs/figures/repeated_seed_pr_auc_boxplot.png

用途：展示不同模型在 10 个学生分组随机划分下的 PR-AUC 稳定性，避免只依赖单次划分结果。

## 图 11 多随机种子窗口 PR-AUC 分布

建议图题：**图 11 XGBoost 在多预测窗口下的重复实验 PR-AUC 分布**

图像路径：outputs/figures/repeated_seed_window_pr_auc.png

用途：展示第 7 天、第 14 天、第 28 天、第 56 天和完整课程窗口在不同随机划分下的性能分布和均值趋势。
