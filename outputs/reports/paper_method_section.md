# 方法部分草稿

## 1. 研究任务与符号定义

本文实验面向在线学习场景中的学业风险早期识别问题。给定学生在某一课程开课学期中的人口统计信息、注册信息、学习平台交互行为以及阶段性评估记录，模型需要预测该学生最终是否属于学业风险群体。实验单位为一个学生-课程-开课学期实例。根据已执行的数据汇总，实验数据包含 32,593 个实例、28,785 名唯一学生和 22 个课程开课学期（outputs/tables/dataset_summary.csv）。

二分类标签由 OULAD 中的 `final_result` 字段构造。若最终结果为 Fail 或 Withdrawn，则定义 `at_risk = 1`；若最终结果为 Pass 或 Distinction，则定义 `at_risk = 0`。执行结果显示，风险类实例数为 17,208，非风险类实例数为 15,385，风险类比例为 0.527966（outputs/tables/dataset_summary.csv）。

## 2. 多窗口早期预测框架

为模拟不同时间点的早期预警，本实验构建了五个预测窗口：第 7 天、第 14 天、第 28 天、第 56 天和完整课程周期。对于任一预测窗口，特征仅由预测日及其之前可观测到的数据构造。完整课程周期窗口用于上界比较，不作为严格早期预警场景解释。

多窗口设置的核心思想是比较模型在不同观测长度下的风险识别能力。第 7 天和第 14 天主要依赖早期 VLE 行为及静态人口统计信息；第 28 天和第 56 天逐渐纳入更多已到期、已提交的评估信息；完整课程周期则使用课程全周期内可用的行为和评估记录。各窗口的执行结果记录于 outputs/tables/time_window_results.csv，并由 outputs/figures/time_window_performance.png 可视化。

## 3. 特征构建

本文构建了人口统计、注册、VLE 行为和评估四类特征。人口统计特征来自 `studentInfo.csv`，包括性别、地区、最高学历、贫困指数分组、年龄段、历史尝试次数、学习学分数和残障状态。注册特征来自 `studentRegistration.csv`，其中 `days_before_start_registered` 仅在注册日期不晚于预测日时可用；若注册日期晚于预测日，则将该特征置为缺失，以避免未来信息泄漏。

VLE 行为特征由 `studentVle.csv` 和 `vle.csv` 构造。对于每个预测窗口，仅保留 `date <= prediction_day` 的交互记录。构造的特征包括预测日前总点击数、活跃天数、平均每活跃日点击数、最大不活跃间隔、访问资源数、按活动类型统计的点击数、访问活动类型数量、周点击趋势斜率、点击熵、首次活动日、预测日前最后活动日、距最后活动日天数，以及活跃天数占可用天数比例。

评估特征由 `studentAssessment.csv` 和 `assessments.csv` 构造。对于每个预测窗口，仅使用评估截止日期不晚于预测日且提交日期不晚于预测日的评估记录。构造的特征包括已提交评估数量、预测日前平均得分、加权平均得分、已完成评估权重、迟交次数、已到期但缺失的评估数量、平均延迟、最大延迟和预测日前最后一次提交评估日期。

## 4. 特征消融设计

为分析不同信息源的贡献，实验设置了五个特征组：仅人口统计特征、仅 VLE 行为特征、仅评估特征、人口统计加 VLE 特征，以及人口统计加 VLE 加评估特征。该设计用于比较静态背景信息、行为参与信息和阶段性学业表现信息在不同预测窗口中的相对贡献。消融实验的完整执行结果保存于 outputs/tables/feature_ablation_results.csv。

## 5. 模型与训练流程

主模型比较在第 56 天预测窗口、人口统计加 VLE 加评估组合特征组上进行。比较模型包括 Logistic Regression、Decision Tree、Random Forest、Linear SVM、XGBoost、LightGBM、CatBoost 和 Stacking Ensemble。模型比较结果保存于 outputs/tables/model_comparison.csv。

所有模型均通过 scikit-learn Pipeline 组织预处理与分类器训练。数值特征使用中位数填补和标准化，类别特征使用众数填补和 one-hot 编码。由于预处理器位于 Pipeline 内部，预处理参数仅在每次训练划分或交叉验证训练折上拟合，而不会在完整数据集上预先拟合。该流程用于降低预处理阶段的数据泄漏风险。

类别不平衡通过模型内部权重机制处理。Logistic Regression、Decision Tree、Linear SVM 和 LightGBM 使用 class weight；Random Forest 使用 balanced subsample；XGBoost 使用由训练集标签计算的 `scale_pos_weight`。实验未使用 SMOTE 或任何在划分前进行的过采样方法。

## 6. 验证协议

主验证协议为按学生标识分组的分层训练-测试划分，表中标记为 `stratified_student_group_train_test`。该划分使用 `id_student` 作为分组变量，以避免同一学生的不同课程记录同时出现在训练集和测试集中。审计结果显示，五个预测窗口的训练集和测试集之间重叠学生数均为 0（outputs/reports/leakage_audit.md）。

除主划分外，实验还进行了按课程开课学期分组的 GroupKFold 验证和 leave-one-course-presentation-out 验证。GroupKFold 使用 5 折，leave-one-course-presentation-out 使用 22 折，二者用于考察跨课程开课学期的泛化表现。相应结果见 outputs/tables/model_comparison.csv。

## 7. 评价指标

实验报告准确率、精确率、召回率、F1、ROC-AUC、PR-AUC、平衡准确率和混淆矩阵。由于风险识别任务存在类别不平衡，仅依赖准确率可能产生误导，因此结果解释优先关注 PR-AUC、F1、平衡准确率以及混淆矩阵。模型比较、时间窗口实验和特征消融实验分别记录于 outputs/tables/model_comparison.csv、outputs/tables/time_window_results.csv 和 outputs/tables/feature_ablation_results.csv。

## 8. 可解释性分析

为解释树模型的预测依据，实验对选定的 XGBoost 模型进行了 SHAP 分析。全局解释结果包括 SHAP summary plot 和 Top 20 平均绝对 SHAP 特征表，分别保存于 outputs/figures/shap_summary.png 和 outputs/tables/shap_top_features.csv。实验还生成了前若干重要特征的依赖图，以及若干正确预测为高风险学生的局部解释，局部解释表保存于 outputs/tables/shap_local_high_risk.csv。

