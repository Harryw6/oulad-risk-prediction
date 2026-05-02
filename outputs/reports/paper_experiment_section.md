# 实验结果

## 1. 数据集描述

本实验使用项目原始数据目录中存放的 Open University Learning Analytics Dataset（OULAD）。已执行的数据集汇总显示，数据集中包含 32,593 个学生-课程-开课学期实例、28,785 名唯一学生，以及 22 个课程开课学期。实验使用的原始表包括：`studentInfo` 表 32,593 行，`studentRegistration` 表 32,593 行，`studentAssessment` 表 173,912 行，`assessments` 表 206 行，`studentVle` 表 10,655,280 行，`vle` 表 6,364 行，以及 `courses` 表 22 行（outputs/tables/dataset_summary.csv）。

二分类标签分布存在中等程度的不平衡。非风险类别包含 15,385 个实例，风险类别包含 17,208 个实例，对应的风险学生比例为 0.527966（outputs/tables/dataset_summary.csv; outputs/figures/label_distribution.png）。从原始最终结果来看，数据集中包含 12,361 个 Pass、3,024 个 Distinction、7,052 个 Fail，以及 10,156 个 Withdrawn 实例（outputs/tables/dataset_summary.csv）。

## 2. 任务定义

预测任务为对学业风险学生进行二分类早期识别。正类定义为 `at_risk = 1`，对应 `final_result` 为 Fail 或 Withdrawn 的学生；负类定义为 `at_risk = 0`，对应 `final_result` 为 Pass 或 Distinction 的学生。每条观测对应一个学生-课程-开课学期实例。目标变量分布及最终结果映射记录于 outputs/tables/dataset_summary.csv。

## 3. 特征工程

特征构建使用了人口统计信息、注册信息、虚拟学习环境（VLE）行为信息和测验/作业评估信息。人口统计特征包括性别、地区、最高学历、贫困指数分组、年龄段、历史尝试次数、学习学分数和残障状态。注册时间仅在预测日之前已经可获得时才被纳入。行为特征汇总了各预测窗口之前的 VLE 活动，包括点击总数、活跃天数、不活跃情况、资源多样性、按活动类型统计的点击数、点击熵以及时间活动指标。评估特征仅使用在预测窗口之前已经到期并已提交的评估记录，包括已提交评估数量、得分汇总、已完成权重、迟交次数、已到期但缺失的评估数量、延迟指标以及最后一次提交评估的日期。

最终特征集被划分为五个消融组：仅人口统计特征、仅 VLE 行为特征、仅评估特征、人口统计特征加 VLE 特征，以及人口统计特征加 VLE 特征加评估特征。这些特征组体现在 outputs/tables/feature_ablation_results.csv 的 `feature_group` 列中。预测窗口包括第 7 天、第 14 天、第 28 天、第 56 天和完整课程周期，相关结果见 outputs/tables/time_window_results.csv 和 outputs/tables/feature_ablation_results.csv。

## 4. 实验设置

主模型比较在第 56 天预测窗口上进行，使用人口统计、VLE 和评估特征的组合特征集。已执行的模型比较表中的验证标记为 `stratified_student_group_train_test`，表示主划分采用按学生标识分组的分层训练-测试协议。该表还报告了课程开课学期层面的验证结果，包括 5 折 `groupkfold_course_presentation` 和 22 折 `leave_one_course_presentation_out`（outputs/tables/model_comparison.csv）。

比较的分类器包括 Logistic Regression、Decision Tree、Random Forest、Linear SVM、XGBoost、LightGBM、CatBoost 和 Stacking Ensemble。结果表为每个模型或验证协议报告了准确率、精确率、召回率、F1、ROC-AUC、PR-AUC、平衡准确率以及混淆矩阵组成部分（outputs/tables/model_comparison.csv）。由于数据集存在类别不平衡，除准确率外，结果解释还重点关注 PR-AUC、F1、平衡准确率和混淆矩阵（outputs/tables/model_comparison.csv）。

## 5. 模型比较结果

在第 56 天、按学生分组的分层训练-测试协议下，CatBoost 在所有比较模型中取得最高 PR-AUC，其 PR-AUC = 0.9186486964962812，ROC-AUC = 0.8919530425452321，F1 = 0.8089641122081178，平衡准确率 = 0.8167538221480547，准确率 = 0.8130081300813008。其混淆矩阵为 TN = 2,719，FP = 358，FN = 861，TP = 2,581（outputs/tables/model_comparison.csv; outputs/figures/confusion_matrix_best_model.png）。

XGBoost 与 CatBoost 在 PR-AUC 上几乎持平，其 PR-AUC = 0.9185809528222727，ROC-AUC = 0.8918490866897415，F1 = 0.8092165898617512，平衡准确率 = 0.812103143092544，准确率 = 0.8094799815922687（outputs/tables/model_comparison.csv）。LightGBM 随后，PR-AUC = 0.9176371906299448，ROC-AUC = 0.890856265781037，F1 = 0.8077942322681216，平衡准确率 = 0.8143065634573545（outputs/tables/model_comparison.csv）。

在非 boosting 基线模型中，Random Forest 的 PR-AUC = 0.9154691788408824，F1 = 0.8067828251400124；Logistic Regression 的 PR-AUC = 0.9119305128388278，F1 = 0.8048019956345495。Linear SVM 的 PR-AUC = 0.9102366416679839，F1 = 0.80296576747121。Decision Tree 基线低于集成模型和线性模型，其 PR-AUC = 0.9001400594123636，F1 = 0.78688（outputs/tables/model_comparison.csv）。

课程开课学期分组验证的结果低于主学生分组训练-测试结果。使用 XGBoost 时，按课程开课学期进行 GroupKFold 得到 PR-AUC = 0.9106072369677266，ROC-AUC = 0.8803302147302228，F1 = 0.7969606110111422，平衡准确率 = 0.7954748459839097，共 5 折。Leave-one-course-presentation-out 评估得到 PR-AUC = 0.868957712485784，ROC-AUC = 0.8663078252476215，F1 = 0.7514300218132678，平衡准确率 = 0.7700311214180374，共 22 折（outputs/tables/model_comparison.csv）。这一下降表明，在更严格的课程开课学期迁移评估下，模型性能弱于主学生分组训练-测试设置。

模型比较的 ROC 曲线和精确率-召回率曲线分别生成于 outputs/figures/roc_curves.png 和 outputs/figures/pr_curves.png。相应的数值指标报告于 outputs/tables/model_comparison.csv。

## 6. 多窗口早期预测结果

多窗口实验使用 XGBoost，并采用人口统计、VLE 和评估特征的组合特征集。随着观测窗口扩展，预测性能逐步提高。在第 7 天，模型得到 PR-AUC = 0.8041911779170687，ROC-AUC = 0.7789676154377372，F1 = 0.7145421903052065，平衡准确率 = 0.7081198115311498，准确率 = 0.7073170731707317。在第 14 天，性能提升至 PR-AUC = 0.8254951083251584，ROC-AUC = 0.7927083417917458，F1 = 0.7155824508320726，平衡准确率 = 0.7130660235818335（outputs/tables/time_window_results.csv; outputs/figures/time_window_performance.png）。

到第 28 天，模型达到 PR-AUC = 0.8832185448773401，ROC-AUC = 0.8503195721966335，F1 = 0.7609740964789825，平衡准确率 = 0.7666357694631138。到第 56 天，模型达到 PR-AUC = 0.9185809528222727，ROC-AUC = 0.8918490866897415，F1 = 0.8092165898617512，平衡准确率 = 0.812103143092544（outputs/tables/time_window_results.csv; outputs/figures/time_window_performance.png）。

完整课程窗口产生最高数值，其中 PR-AUC = 0.9894895678678277，ROC-AUC = 0.985792227652182，F1 = 0.9474157303370786，平衡准确率 = 0.9477887617016432，准确率 = 0.9461573861021629（outputs/tables/time_window_results.csv）。该完整课程结果应解释为上界比较，而不是早期预警结果，因为它使用了完整可用课程周期的数据。

## 7. 特征消融结果

特征消融实验表明，仅使用人口统计特征在所有窗口上的性能相对有限。例如，仅人口统计特征的 PR-AUC 在第 7 天为 0.6711683980597183，第 14 天为 0.6714085190813539，第 28 天为 0.6718281289333156，第 56 天为 0.6707936428224462，完整课程设置下为 0.6721860046123783（outputs/tables/feature_ablation_results.csv）。

VLE 行为特征在所有窗口上均优于仅人口统计特征。仅 VLE 特征的 PR-AUC 从第 7 天的 0.7579831232784434 增加到第 14 天的 0.7906172551642031、第 28 天的 0.8286905118028831、第 56 天的 0.8757945592173886，并在完整课程窗口达到 0.9775490616226298（outputs/tables/feature_ablation_results.csv）。

仅评估特征在最早期窗口表现较弱，但随着评估信息积累而变强。在第 7 天，仅评估特征的 ROC-AUC 为 0.5，PR-AUC 为 0.5279950912716674，混淆矩阵显示 TN = 0，FP = 3,077，FN = 0，TP = 3,442。到第 28 天，仅评估特征的 PR-AUC 提升至 0.8361077812002313，到第 56 天达到 PR-AUC = 0.9019258000234883。在完整课程设置下，仅评估特征的 PR-AUC 为 0.9873322850982235（outputs/tables/feature_ablation_results.csv）。

在部分早期窗口中，人口统计特征与 VLE 特征组合优于仅 VLE 特征。在第 7 天，人口统计加 VLE 的 PR-AUC = 0.8037004457739649，而仅 VLE 的 PR-AUC = 0.7579831232784434。在第 14 天，人口统计加 VLE 的 PR-AUC = 0.824310163656357，而仅 VLE 的 PR-AUC = 0.7906172551642031。在第 56 天，人口统计加 VLE 的 PR-AUC = 0.8882076116219266，而仅 VLE 的 PR-AUC = 0.8757945592173886（outputs/tables/feature_ablation_results.csv）。

人口统计、VLE 与评估的组合特征组在第 7 天、第 28 天、第 56 天和完整课程窗口取得最佳 PR-AUC。其 PR-AUC 在第 7 天为 0.8041911779170687，第 14 天为 0.8254951083251584，第 28 天为 0.8832185448773401，第 56 天为 0.9185809528222727，完整课程窗口为 0.9894895678678277。在第 14 天，人口统计加 VLE 的 PR-AUC = 0.824310163656357，组合特征集的 PR-AUC = 0.8254951083251584，表明已执行结果中二者差异很小（outputs/tables/feature_ablation_results.csv）。

## 8. SHAP 可解释性结果

针对所选 XGBoost 模型进行了 SHAP 分析。全局 SHAP 汇总图生成于 outputs/figures/shap_summary.png，Top 特征表见 outputs/tables/shap_top_features.csv。最高的平均绝对 SHAP 值对应 `vle_days_since_last_activity`，其 mean absolute SHAP = 0.7940888。第二高的特征为 `assessment_missing_due_count`，mean absolute SHAP = 0.7315784。随后最重要的特征包括 `assessment_mean_score`，其值为 0.27383345；`assessment_weighted_mean_score`，其值为 0.19731991；以及 `assessment_completed_weight`，其值为 0.19602647（outputs/tables/shap_top_features.csv; outputs/figures/shap_summary.png）。

其他排名较高的特征包括 `assessment_due_weight`，mean absolute SHAP = 0.1803406；`assessment_average_delay`，其值为 0.14879794；`highest_education_Lower Than A Level`，其值为 0.14576882；`vle_clicks_activity_page`，其值为 0.1051074；以及 `vle_active_day_ratio`，其值为 0.100095406（outputs/tables/shap_top_features.csv）。针对主要 SHAP 特征生成了依赖图，包括 outputs/figures/shap_dependence_vle_days_since_last_activity.png、outputs/figures/shap_dependence_assessment_missing_due_count.png、outputs/figures/shap_dependence_assessment_mean_score.png、outputs/figures/shap_dependence_assessment_weighted_mean_score.png 和 outputs/figures/shap_dependence_assessment_completed_weight.png。

此外，还为正确预测为高风险的学生生成了局部解释。例如，课程开课学期 AAA 2013J 中学生 30268 的一个局部解释显示，绝对 SHAP 贡献最大的特征为 `assessment_missing_due_count`，SHAP 值为 2.198883056640625，以及 `vle_days_since_last_activity`，SHAP 值为 1.9774171113967896。AAA 2013J 中学生 305539 的另一个局部解释为 `assessment_missing_due_count` 分配了 2.1509902477264404 的 SHAP 值，为 `vle_days_since_last_activity` 分配了 2.1366493701934814 的 SHAP 值（outputs/tables/shap_local_high_risk.csv）。

## 9. 教育解释

已执行结果表明，早期预警性能强烈依赖可观测学习证据的数量和类型。在第 7 天和第 14 天，VLE 行为和人口统计信息提供了大部分可用信号，而仅评估特征较为有限；例如，仅评估特征在第 7 天的 PR-AUC = 0.5279950912716674，在第 14 天的 PR-AUC = 0.5465226996626236（outputs/tables/feature_ablation_results.csv）。这与课程早期评估结果可用性有限的情况一致。

随着课程推进，评估特征变得越来越有信息量。仅评估特征的 PR-AUC 在第 28 天提升至 0.8361077812002313，在第 56 天提升至 0.9019258000234883（outputs/tables/feature_ablation_results.csv）。组合模型也从第 7 天的 PR-AUC = 0.8041911779170687 提升至第 56 天的 PR-AUC = 0.9185809528222727（outputs/tables/time_window_results.csv）。这些结果表明，行为参与信号从课程初期即具有作用，而当评估到期和提交记录逐渐可用后，评估证据的重要性不断增强。

SHAP 结果支持这一解释。排名最高的两个特征分别为 `vle_days_since_last_activity` 和 `assessment_missing_due_count`，其平均绝对 SHAP 值分别为 0.7940888 和 0.7315784（outputs/tables/shap_top_features.csv）。从教育意义上看，这些变量分别对应学习平台参与度下降以及未完成已到期评估活动。得分相关特征的突出表现，包括 `assessment_mean_score` 和 `assessment_weighted_mean_score`，进一步表明当评估可用后，已观察到的学业表现会对模型产生贡献（outputs/tables/shap_top_features.csv）。

## 10. 稳健性与统计检验补充

为增强单次划分结果的可信度，本项目进一步生成了 bootstrap 置信区间、退课样本敏感性分析、校准与阈值分析，以及多随机种子重复实验。第 56 天 XGBoost 的 PR-AUC 为 0.9185809528222727，bootstrap 95% 置信区间为 [0.9119457341880745, 0.9248238976274022]；F1 为 0.8092165898617512，95% 置信区间为 [0.7984249963478957, 0.8191773160603579]（outputs/tables/bootstrap_ci_time_window_results.csv）。

多随机种子重复实验使用 10 个学生分组随机划分种子。第 56 天模型比较中，XGBoost 的平均 PR-AUC 为 0.9181163962207144，标准差为 0.0020750775379833543；LightGBM 的平均 PR-AUC 为 0.9176360017687116；CatBoost 的平均 PR-AUC 为 0.917397192442683（outputs/tables/repeated_seed_model_summary.csv; outputs/figures/repeated_seed_pr_auc_boxplot.png）。在多窗口重复实验中，XGBoost 的平均 PR-AUC 从第 7 天的 0.7992976125065154 提升到第 56 天的 0.9181163962207144，完整课程窗口达到 0.9891930863639258（outputs/tables/repeated_seed_window_summary.csv; outputs/figures/repeated_seed_window_pr_auc.png）。

配对 Wilcoxon signed-rank 检验和 Holm 校正结果显示，第 56 天相对第 7 天、第 14 天和第 28 天的 PR-AUC、F1、ROC-AUC 与平衡准确率提升在 10 个随机划分上均达到校正后 p < 0.05（outputs/tables/significance_tests.csv）。模型比较中，XGBoost 相对 Random Forest、Logistic Regression、Linear SVM 和 Decision Tree 的 PR-AUC 差异达到校正后 p < 0.05；相对 LightGBM 的 PR-AUC 差异未达到校正后 p < 0.05（outputs/tables/significance_tests.csv）。因此，论文中应将 XGBoost、LightGBM 和 CatBoost 的性能差异表述为数值接近，而不是夸大为明显优势。

校准分析显示，第 56 天 XGBoost 的 Brier score 为 0.13042496144771576，10-bin ECE 为 0.010918549872154201（outputs/tables/calibration_summary_day56.csv; outputs/figures/calibration_curve_day56.png）。阈值分析表明，阈值为 0.10 时召回率为 0.9933178384660082、精确率为 0.561965811965812；阈值为 0.90 时精确率为 0.984036488027366、召回率为 0.5014526438117374（outputs/tables/threshold_analysis_day56.csv; outputs/figures/threshold_tradeoff_day56.png）。这些结果可用于讨论不同教育干预资源约束下的预警阈值选择。

## 11. 局限性

在解释这些结果时应考虑若干局限性。首先，完整课程窗口取得了最高性能，其中 PR-AUC = 0.9894895678678277，F1 = 0.9474157303370786，但该设置不是早期预警场景，应仅作为上界比较（outputs/tables/time_window_results.csv）。

其次，跨课程开课学期的泛化能力弱于主学生分组划分。在学生分组训练-测试划分下，第 56 天 XGBoost 的 PR-AUC 为 0.9185809528222727，而按课程开课学期的 GroupKFold 得到 PR-AUC = 0.9106072369677266，leave-one-course-presentation-out 得到 PR-AUC = 0.868957712485784（outputs/tables/model_comparison.csv）。这表明跨课程开课学期迁移比随机学生分组评估更困难。

第三，重复实验改变的是训练-测试划分随机种子，模型内部随机状态和超参数仍沿用主实验固定配置。因此，重复实验主要反映数据划分方差，不能替代嵌套交叉验证或系统超参数优化（outputs/tables/repeated_seed_model_summary.csv; outputs/tables/significance_tests.csv）。

第四，排除预测日前已退课样本后的敏感性分析显示，第 56 天 PR-AUC 下降至 0.8136050880203111，说明 Withdrawn 样本对任务难度和性能解释有重要影响（outputs/tables/withdrawal_sensitivity_results.csv）。论文中应明确主任务预测的是 Fail 或 Withdrawn 的综合风险，而不是只预测最终考试失败。

第五，SHAP 分析基于已生成的 SHAP 表格和图形进行报告，但其解释仍然是模型特定的。全局和局部 SHAP 输出识别了有影响力的模型特征，但并不建立学习者行为与学业结果之间的因果关系（outputs/tables/shap_top_features.csv; outputs/tables/shap_local_high_risk.csv; outputs/figures/shap_summary.png）。
