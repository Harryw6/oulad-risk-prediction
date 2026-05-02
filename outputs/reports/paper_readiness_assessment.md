# EI 投稿准备度评估与补充实验

## 1. 结论

在现有实验基础上，本项目已经具备撰写一篇 EI 会议或 EI 期刊投稿实验部分的基本完整性。当前结果包含数据泄漏审计、多模型比较、多窗口早期预测、特征消融、SHAP 解释、跨课程开课学期验证、bootstrap 置信区间、退课样本敏感性分析、校准与阈值分析，以及多随机种子重复实验和配对统计显著性检验。

因此，当前版本可以支撑一版较完整的投稿稿件。但论文写作时仍应保持审慎：不要声称模型达到最优，不要把 SHAP 解释写成因果解释，也不要把完整课程窗口写成早期预警结果。若目标是冲击更高水平的期刊，后续仍建议增加系统超参数优化、外部数据验证或更细粒度的课程异质性分析。

## 2. 新增结果文件

- outputs/tables/bootstrap_ci_time_window_results.csv：多窗口指标 bootstrap 95% 置信区间。
- outputs/tables/per_course_presentation_results.csv：逐课程开课学期 leave-one-course-presentation-out 明细。
- outputs/tables/withdrawal_sensitivity_results.csv：排除预测日前已退课样本后的敏感性分析。
- outputs/tables/calibration_summary_day56.csv：第 56 天 XGBoost 校准汇总。
- outputs/tables/calibration_bins_day56.csv：第 56 天校准分箱明细。
- outputs/tables/threshold_analysis_day56.csv：第 56 天不同阈值下的 precision、recall、F1 和 balanced accuracy。
- outputs/tables/repeated_seed_model_results.csv：第 56 天模型比较的逐 seed 结果。
- outputs/tables/repeated_seed_model_summary.csv：第 56 天模型比较的均值、标准差、最小值和最大值。
- outputs/tables/repeated_seed_window_results.csv：XGBoost 多窗口逐 seed 结果。
- outputs/tables/repeated_seed_window_summary.csv：XGBoost 多窗口重复实验汇总。
- outputs/tables/significance_tests.csv：配对 Wilcoxon signed-rank 检验和 Holm 校正 p 值。
- outputs/figures/calibration_curve_day56.png：校准曲线。
- outputs/figures/threshold_tradeoff_day56.png：阈值权衡曲线。
- outputs/figures/repeated_seed_pr_auc_boxplot.png：多随机种子模型 PR-AUC 分布。
- outputs/figures/repeated_seed_window_pr_auc.png：多随机种子窗口 PR-AUC 分布。
- outputs/reports/paper_significance_section.md：可写入论文的中文重复实验与统计检验段落。

## 3. 关键补充发现

第 56 天 XGBoost 的 PR-AUC 为 0.9185809528222727，bootstrap 95% CI 为 [0.9119457341880745, 0.9248238976274022]；F1 为 0.8092165898617512，95% CI 为 [0.7984249963478957, 0.8191773160603579]（outputs/tables/bootstrap_ci_time_window_results.csv）。

逐课程开课学期留一验证显示，不同课程开课学期之间存在明显性能差异。PR-AUC 的均值为 0.868957712485784，最小值为 0.5929267244446996，最大值为 0.9641690908471405（outputs/tables/per_course_presentation_results.csv）。这说明跨课程泛化并不均匀，论文中应避免只报告总体均值。

在第 56 天排除预测日前已退课样本后，样本量从 32,593 变为 26,522，排除 6,071 个实例；该敏感性设置下 PR-AUC = 0.8136050880203111，F1 = 0.7090362837993991（outputs/tables/withdrawal_sensitivity_results.csv）。该结果应作为对 Withdrawn 标签影响的稳健性补充。

第 56 天 XGBoost 的 Brier score 为 0.13042496144771576，10-bin ECE 为 0.010918549872154201（outputs/tables/calibration_summary_day56.csv; outputs/figures/calibration_curve_day56.png）。阈值分析显示，低阈值提高召回率，高阈值提高精确率，相关结果保存在 outputs/tables/threshold_analysis_day56.csv。

多随机种子重复实验显示，第 56 天窗口下 XGBoost 的平均 PR-AUC 为 0.9181163962207144，标准差为 0.0020750775379833543；LightGBM 的平均 PR-AUC 为 0.9176360017687116；CatBoost 的平均 PR-AUC 为 0.917397192442683（outputs/tables/repeated_seed_model_summary.csv）。配对 Wilcoxon 检验显示，XGBoost 相对 LightGBM 的 PR-AUC 差异在 Holm 校正后未达到 p < 0.05，而相对 Random Forest、Logistic Regression、Linear SVM 和 Decision Tree 的 PR-AUC 差异达到 p < 0.05（outputs/tables/significance_tests.csv）。

多窗口重复实验显示，XGBoost 的平均 PR-AUC 从第 7 天的 0.7992976125065154 提升到第 14 天的 0.8209529361685834、第 28 天的 0.878571321033952、第 56 天的 0.9181163962207144；完整课程窗口达到 0.9891930863639258（outputs/tables/repeated_seed_window_summary.csv）。第 56 天相对第 7 天、第 14 天和第 28 天的 PR-AUC、F1、ROC-AUC 与 balanced accuracy 提升在 Holm 校正后均达到 p < 0.05（outputs/tables/significance_tests.csv）。

## 4. 对论文写作的影响

建议在论文实验部分设置以下小节：数据集与任务定义、多窗口特征工程、模型比较、多窗口早期预测、特征消融、跨课程泛化、稳健性与敏感性分析、校准与阈值分析、重复实验与统计检验、SHAP 可解释性分析。这样可以将工作从单次性能报告提升为较完整的实验评估。

核心结论应写得克制：结果支持“随着观测窗口延长，风险预测性能显著提升；VLE 行为在早期窗口中重要，评估特征在第 28 天后贡献增强；XGBoost/LightGBM/CatBoost 在第 56 天表现接近”。不建议写成“某一个模型绝对优于所有模型”。

## 5. 仍需谨慎表述的内容

- 不应声称模型具有因果解释能力；SHAP 仅说明模型关联性贡献。
- 完整课程窗口只能作为性能上界，不是早期预警结果。
- 当前未进行系统超参数搜索，模型设置以可复现和合理运行时间为主。
- 重复实验主要反映不同学生分组随机划分造成的方差，模型内部随机状态仍沿用固定配置。
- 排除预测日前已退课样本后性能明显下降，论文中必须说明 Withdrawn 标签会影响任务难度和解释。
