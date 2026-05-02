# 稳健性、泛化与部署分析

## 1. Bootstrap 置信区间分析

为评估多窗口预测结果的统计稳定性，本研究对 XGBoost 在各预测窗口下的测试集预测结果进行了 bootstrap 置信区间估计。每个窗口使用 1,000 次 bootstrap 重采样，并报告主要评价指标的 95% 置信区间。完整结果见 outputs/tables/bootstrap_ci_time_window_results.csv。

在第 7 天窗口，XGBoost 的 PR-AUC 为 0.8041911779170687，95% 置信区间为 [0.7910695012228276, 0.8168047704591246]；F1 为 0.7145421903052065，95% 置信区间为 [0.7022056179587656, 0.7259766915408066]。在第 14 天窗口，PR-AUC 为 0.8254951083251584，95% 置信区间为 [0.8138834481660749, 0.8368298245608112]；F1 为 0.7155824508320726，95% 置信区间为 [0.702852244571895, 0.7272471169341086]（outputs/tables/bootstrap_ci_time_window_results.csv）。

随着观测窗口延长，性能及其置信区间整体上移。在第 28 天窗口，PR-AUC 为 0.8832185448773401，95% 置信区间为 [0.8741604549244661, 0.8911972267040089]；F1 为 0.7609740964789825，95% 置信区间为 [0.749128419189177, 0.7719738256393878]。在第 56 天窗口，PR-AUC 为 0.9185809528222727，95% 置信区间为 [0.9119457341880745, 0.9248238976274022]；F1 为 0.8092165898617512，95% 置信区间为 [0.7984249963478957, 0.8191773160603579]（outputs/tables/bootstrap_ci_time_window_results.csv）。

完整课程窗口取得最高性能，PR-AUC 为 0.9894895678678277，95% 置信区间为 [0.98776800812518, 0.9910948236105134]；F1 为 0.9474157303370786，95% 置信区间为 [0.9416235649836533, 0.9525981756652138]（outputs/tables/bootstrap_ci_time_window_results.csv）。该结果仍应解释为全周期上界，而非严格早期预警性能。

## 2. 逐课程开课学期泛化分析

除总体 leave-one-course-presentation-out 结果外，本研究进一步生成了逐课程开课学期的留一验证明细，以考察模型跨课程开课学期迁移时的异质性。完整结果见 outputs/tables/per_course_presentation_results.csv。

逐课程开课学期结果显示，PR-AUC 在不同课程开课学期间差异明显。22 个留一测试课程开课学期的 PR-AUC 均值为 0.868957712485784，最小值为 0.5929267244446996，最大值为 0.9641690908471405（outputs/tables/per_course_presentation_results.csv）。其中 CCC_2014B 的 PR-AUC 最高，为 0.9641690908471405；GGG_2013J 的 PR-AUC 最低，为 0.5929267244446996（outputs/tables/per_course_presentation_results.csv）。

这种差异说明，虽然总体 leave-one-course-presentation-out 的 PR-AUC 为 0.868957712485784，但不同课程开课学期之间的模型迁移难度并不一致。论文中不应仅报告总体均值，还应说明跨课程泛化存在课程层面的性能波动（outputs/tables/per_course_presentation_results.csv）。

## 3. 退课样本敏感性分析

由于目标变量将 Withdrawn 归为风险类，部分学生可能在预测窗口之前已经退课。虽然 `date_unregistration` 未作为模型特征使用，但已退课学生的后续不活跃行为可能影响早期预警任务解释。为此，本研究进行了敏感性分析：在每个预测窗口中，排除 `date_unregistration <= prediction_day` 的学生实例，然后重新训练并评估 XGBoost。完整结果见 outputs/tables/withdrawal_sensitivity_results.csv。

敏感性分析显示，排除预测日前已退课样本后，任务难度明显增加。在第 7 天窗口，样本量从 32,593 降至 29,178，排除 3,415 个实例；PR-AUC 为 0.6950092250603459，F1 为 0.6477908025247971。第 14 天窗口中，样本量降至 28,119，排除 4,474 个实例；PR-AUC 为 0.702505748413817，F1 为 0.6509988249118684（outputs/tables/withdrawal_sensitivity_results.csv）。

在第 28 天窗口，排除后样本量为 27,538，排除 5,055 个实例；PR-AUC 为 0.7720474324159741，F1 为 0.6748155953635405。在第 56 天窗口，排除后样本量为 26,522，排除 6,071 个实例；PR-AUC 为 0.8136050880203111，F1 为 0.7090362837993991（outputs/tables/withdrawal_sensitivity_results.csv）。

与原始第 56 天结果 PR-AUC = 0.9185809528222727 相比，排除预测日前已退课样本后的第 56 天 PR-AUC = 0.8136050880203111，说明原始任务中的 Withdrawn 样本确实对模型性能产生较大影响（outputs/tables/time_window_results.csv; outputs/tables/withdrawal_sensitivity_results.csv）。因此，论文中应将该敏感性分析作为有效性讨论的重要补充。

## 4. 校准分析

除判别性能外，本研究还评估了第 56 天 XGBoost 模型的概率校准情况。校准汇总结果显示，模型 Brier score 为 0.13042496144771576，10-bin expected calibration error（ECE）为 0.010918549872154201，平均预测风险为 0.5314969420433044，测试集实际风险比例为 0.5279950912716674（outputs/tables/calibration_summary_day56.csv）。

校准曲线见 outputs/figures/calibration_curve_day56.png，分箱明细见 outputs/tables/calibration_bins_day56.csv。该结果表明，在当前测试集上，模型平均预测风险与实际风险比例较为接近。不过，校准分析仅在第 56 天 XGBoost 模型上执行，尚未扩展到所有模型和所有预测窗口。

## 5. 决策阈值分析

实际预警系统通常需要根据干预资源选择不同决策阈值。因此，本研究对第 56 天 XGBoost 模型进行了阈值分析，阈值范围为 0.10 到 0.90。完整结果保存于 outputs/tables/threshold_analysis_day56.csv，可视化结果见 outputs/figures/threshold_tradeoff_day56.png。

阈值分析显示，较低阈值可以提高召回率，但会降低精确率。例如阈值为 0.10 时，召回率为 0.9933178384660082，精确率为 0.561965811965812，F1 为 0.7178249002729372。阈值为 0.50 时，召回率为 0.7652527600232423，精确率为 0.8585397653194263，F1 为 0.8092165898617512。阈值为 0.90 时，精确率提高至 0.984036488027366，但召回率降至 0.5014526438117374（outputs/tables/threshold_analysis_day56.csv）。

该结果表明，若教学管理目标是尽可能发现风险学生，可采用较低阈值以提高召回率；若目标是减少误报并集中有限干预资源，则可采用较高阈值以提高精确率。论文中应明确阈值选择取决于实际教育干预场景，而不是仅依赖默认 0.50 阈值。

## 6. 小结

新增稳健性分析表明，原有实验结论在 bootstrap 置信区间下具有一定稳定性，但跨课程开课学期泛化存在显著异质性；同时，排除预测日前已退课学生后，模型性能明显下降，说明退课相关样本对任务难度和性能解释具有重要影响。校准和阈值分析进一步表明，该模型可以从判别性能角度扩展到实际预警策略讨论。

多随机种子重复实验和统计显著性检验已经补充到 outputs/reports/paper_significance_section.md。该新增结果显示，第 56 天 XGBoost 在 10 个学生分组随机划分下的平均 PR-AUC 为 0.9181163962207144，标准差为 0.0020750775379833543；第 56 天相对第 7 天、第 14 天和第 28 天的 PR-AUC、F1、ROC-AUC 与 balanced accuracy 提升在 Holm 校正后均达到 p < 0.05（outputs/tables/repeated_seed_window_summary.csv; outputs/tables/significance_tests.csv）。后续若继续增强论文，可优先考虑更系统的超参数优化、外部数据验证和更细粒度的课程异质性建模。
