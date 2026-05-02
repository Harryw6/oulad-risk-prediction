# OULAD 学业风险预测项目通俗报告

## 1. 我们做了什么

这个项目围绕一个教育数据挖掘问题展开：在在线学习课程还没有结束时，能不能根据学生已经产生的学习行为，提前判断哪些学生可能最终挂科或退课。

我们使用的数据集是 Open University Learning Analytics Dataset（OULAD）。这是一个开放大学在线学习数据集，包含学生基本信息、课程信息、注册信息、作业/测验记录，以及学生在虚拟学习环境（VLE）中的点击行为。实验中一共使用了 32,593 条“学生-课程-开课学期”记录，涉及 28,785 名学生和 22 个课程开课学期。标签定义很直接：最终结果是 Fail 或 Withdrawn 的学生记为风险学生，最终结果是 Pass 或 Distinction 的学生记为非风险学生。风险学生共有 17,208 条记录，非风险学生共有 15,385 条记录（outputs/tables/dataset_summary.csv）。

我们最终完成的是一套可复现实验流水线，而不是只跑一个模型。项目包括数据加载、特征构造、模型训练、评估、泄漏审计、稳健性分析、SHAP 可解释性分析，以及面向论文写作的中英文结果报告。完整代码和结果已经整理在 GitHub 仓库中。

## 2. 文章构思是什么

这篇文章的核心思路可以概括为一句话：

在在线课程中，越早发现风险学生，教师和平台就越早有机会干预；但越早预测，可用信息越少，所以需要设计多时间窗口的早期预警框架，并解释模型为什么做出这些判断。

因此，文章可以围绕三个问题展开：

1. 在课程早期，例如第 7 天、第 14 天、第 28 天、第 56 天，模型能多准确地预测风险学生？
2. 哪些信息最有用：学生人口统计信息、VLE 学习行为、还是作业/测验表现？
3. 模型做出判断时主要依赖哪些具体信号，这些信号是否能给教学干预提供解释？

对应的文章结构可以这样设计：

- 引言：说明在线学习中学生掉队和退课是重要问题，早期预警有实际意义。
- 相关工作：介绍学习分析、在线学习风险预测、行为特征和可解释机器学习。
- 方法：提出多窗口早期预警框架，分别在第 7、14、28、56 天和完整课程窗口构造特征。
- 实验：比较多种机器学习模型，包括 Logistic Regression、Decision Tree、Random Forest、Linear SVM、XGBoost、LightGBM、CatBoost 和 Stacking。
- 结果：报告模型比较、多窗口预测、特征消融、跨课程验证、稳健性检验和 SHAP 解释。
- 讨论：解释教育意义、实际干预阈值、数据泄漏控制和局限性。

文章的贡献不要写得过大。更稳妥的表述是：本文实现并评估了一个基于 OULAD 的多窗口早期预警实验框架，比较不同模型和特征来源，并结合 SHAP 分析解释模型预测依据。

## 3. 我们如何避免“提前偷看答案”

这个任务最容易出错的地方是数据泄漏。比如，如果模型在第 7 天预测，却用了第 56 天之后的点击数据、未来作业分数，或者直接用了最终成绩字段，那结果就会虚高，没有学术可信度。

本项目专门做了泄漏控制：

- `final_result` 只用于生成标签，不作为模型特征。
- `id_student` 只用于分组和关联，不作为模型输入。
- VLE 点击数据只使用预测日当天及之前的数据。
- 作业/测验特征只使用预测日前已经到期并且已经提交的记录。
- 退课日期 `date_unregistration` 没有作为预测特征。
- 预处理放在 sklearn pipeline 中，只在训练集上拟合。
- 主训练测试划分按学生分组，避免同一个学生同时出现在训练集和测试集中。

泄漏审计报告见 outputs/reports/leakage_audit.md。

## 4. 实验怎么做

我们为每个学生-课程实例构造一行特征。特征主要分三类：

- 人口统计特征：性别、地区、最高学历、年龄段、学分数、是否残障等。
- VLE 行为特征：点击总数、活跃天数、平均点击、最长不活跃间隔、访问资源种类、按活动类型统计的点击数、点击趋势、点击熵等。
- 评估特征：已提交作业数量、平均分、加权平均分、完成权重、迟交次数、缺失已到期作业数量、提交延迟等。

我们设置了五个预测窗口：第 7 天、第 14 天、第 28 天、第 56 天，以及完整课程窗口。完整课程窗口不是早期预警结果，只作为性能上限参考。

评估指标包括 Accuracy、Precision、Recall、F1、ROC-AUC、PR-AUC、Balanced Accuracy 和混淆矩阵。由于风险学生和非风险学生比例并不完全均衡，报告中特别关注 PR-AUC、F1 和 Balanced Accuracy。

## 5. 模型比较结果

在第 56 天主实验中，CatBoost 的单次划分 PR-AUC 最高，为 0.918649；XGBoost 的 PR-AUC 为 0.918581，LightGBM 为 0.917637，三者非常接近（outputs/tables/model_comparison.csv）。

XGBoost 的第 56 天结果为：

- Accuracy = 0.809480
- Precision = 0.858540
- Recall = 0.765253
- F1 = 0.809217
- ROC-AUC = 0.891849
- PR-AUC = 0.918581

从通俗角度看，这说明模型已经能够较好地区分风险学生和非风险学生。Precision 较高表示：被模型判为风险的学生中，相当一部分确实属于风险类别。Recall 为 0.765253 表示：真实风险学生中，模型能够找出大约四分之三。

不过，重复实验告诉我们，不能简单说 XGBoost 明显超过所有模型。10 个随机划分下，XGBoost 平均 PR-AUC 为 0.918116，LightGBM 为 0.917636，CatBoost 为 0.917397（outputs/tables/repeated_seed_model_summary.csv）。XGBoost 相对 LightGBM 的 PR-AUC 差异在 Holm 校正后 p = 0.064453，没有达到 0.05 显著性水平（outputs/tables/significance_tests.csv）。因此论文里更稳妥的说法是：XGBoost、LightGBM 和 CatBoost 在第 56 天都表现很好，XGBoost 在重复实验中的平均值略高，但与 LightGBM 的差异不显著。

## 6. 多窗口早期预测结果

多窗口结果非常符合直觉：课程进行得越久，可用信息越多，模型预测越准确。

XGBoost 的 PR-AUC 从第 7 天的 0.804191，提高到第 14 天的 0.825495、第 28 天的 0.883219、第 56 天的 0.918581。完整课程窗口达到 0.989490，但这只能作为上限参考，不能当作早期预警结果（outputs/tables/time_window_results.csv）。

对应的 F1 也逐步提高：

- 第 7 天：0.714542
- 第 14 天：0.715582
- 第 28 天：0.760974
- 第 56 天：0.809217
- 完整课程：0.947416

重复实验进一步验证了这个趋势。10 个随机划分下，XGBoost 的平均 PR-AUC 从第 7 天的 0.799298 提升到第 56 天的 0.918116。第 56 天相对第 7 天和第 28 天的 PR-AUC 提升在 Holm 校正后都达到 p = 0.005859（outputs/tables/repeated_seed_window_summary.csv; outputs/tables/significance_tests.csv）。

这可以成为文章的重要结论：多窗口设置能够清楚展示早期预警性能随时间增长的过程，而不是只给出一个单点结果。

## 7. 哪些特征最有用

特征消融实验回答了一个关键问题：到底是学生背景、学习行为，还是作业成绩更重要？

结果显示，仅人口统计特征的效果有限，第 56 天 PR-AUC 只有 0.670794。仅 VLE 行为特征在第 56 天 PR-AUC 为 0.875795。仅评估特征在第 56 天 PR-AUC 为 0.901926。把人口统计、VLE 和评估特征合在一起，第 56 天 PR-AUC 达到 0.918581（outputs/tables/feature_ablation_results.csv）。

这说明：

- 课程很早期时，VLE 行为尤其重要，因为作业/测验信息还不充分。
- 到第 28 天和第 56 天后，评估特征开始变得非常有用。
- 最好的结果来自多种信息的组合，而不是只看某一类数据。

第 7 天时，仅评估特征 PR-AUC 只有 0.527995，接近较弱水平；但到第 56 天，仅评估特征 PR-AUC 提升到 0.901926。这一点很适合在论文中解释：课程早期缺少作业表现信息，所以行为数据更关键；课程推进后，作业提交和得分会显著增强模型判断能力。

## 8. 模型主要看什么

SHAP 可解释性分析显示，模型最关注的两个因素是：

1. `vle_days_since_last_activity`：距离上次学习平台活动过去了多少天。
2. `assessment_missing_due_count`：已经到期但缺失的评估数量。

这两个特征的 mean absolute SHAP 分别为 0.794089 和 0.731578，是最重要的两个特征（outputs/tables/shap_top_features.csv）。

通俗解释就是：如果一个学生很久没有登录或使用学习平台，并且已经有到期作业没有完成，那么模型会更倾向于认为这个学生有风险。

其他重要特征还包括平均成绩、加权平均成绩、已完成评估权重、评估迟交情况、最高学历以及页面点击行为。这些结果比较符合教育直觉：学习参与度下降、作业缺失、成绩较低和迟交行为，都可能是风险信号。

SHAP 图和表格见：

- outputs/tables/shap_top_features.csv
- outputs/figures/shap_summary.png
- outputs/figures/shap_dependence_vle_days_since_last_activity.png
- outputs/figures/shap_dependence_assessment_missing_due_count.png

## 9. 结果对教育场景有什么意义

这个实验的教育意义主要有三点。

第一，早期预警是可行的。即使只看第 7 天数据，XGBoost 的 PR-AUC 也达到 0.804191，说明课程非常早期已经存在可用信号（outputs/tables/time_window_results.csv）。

第二，预警不应该只看成绩。第 7 天和第 14 天时，很多作业成绩还没有产生，VLE 行为更能反映学生是否开始参与课程。因此平台点击、活跃天数、不活跃间隔等行为指标很重要。

第三，模型结果可以帮助设计干预策略。例如，对于很久没有活动、缺失已到期作业的学生，教师或平台可以优先发送提醒、提供辅导资源，或安排人工跟进。

不过，模型只能提供风险提示，不能替代教师判断。SHAP 解释也只是模型层面的关联解释，不能说明某个行为一定导致挂科或退课。

## 10. 稳健性和可信度

除了主实验，我们还做了多种补充分析。

Bootstrap 置信区间显示，第 56 天 XGBoost 的 PR-AUC 为 0.918581，95% 置信区间为 [0.911946, 0.924824]；F1 为 0.809217，95% 置信区间为 [0.798425, 0.819177]（outputs/tables/bootstrap_ci_time_window_results.csv）。

跨课程开课学期验证显示，模型迁移到不同课程时性能会下降。Leave-one-course-presentation-out 的平均 PR-AUC 为 0.868958，低于主学生分组划分结果（outputs/tables/model_comparison.csv; outputs/tables/per_course_presentation_results.csv）。这说明模型不是在所有课程上都同样强，论文里需要如实讨论课程差异。

退课样本敏感性分析显示，如果排除预测日前已经退课的学生，第 56 天 PR-AUC 降到 0.813605，F1 降到 0.709036（outputs/tables/withdrawal_sensitivity_results.csv）。这说明 Withdrawn 标签对任务有重要影响。论文应明确本研究预测的是“Fail 或 Withdrawn 的综合风险”，不是单纯预测挂科。

校准和阈值分析也已经完成。第 56 天 XGBoost 的 Brier score 为 0.130425，10-bin ECE 为 0.010919（outputs/tables/calibration_summary_day56.csv）。阈值分析显示，如果把预警阈值设得很低，可以找到更多风险学生，但误报也会增加；如果阈值设得很高，误报减少，但会漏掉更多风险学生（outputs/tables/threshold_analysis_day56.csv）。

## 11. 文章可以怎么写结论

这篇文章可以形成以下结论：

1. 基于 OULAD 的在线学习数据，可以构建较有效的学业风险早期预警模型。
2. 多窗口实验显示，随着课程推进，模型性能逐步提高；第 56 天明显优于第 7 天、第 14 天和第 28 天。
3. 行为数据在课程早期尤其重要，评估数据在课程推进后贡献显著增强。
4. XGBoost、LightGBM 和 CatBoost 都表现较好，其中 XGBoost 在重复实验中平均 PR-AUC 最高，但相对 LightGBM 的 PR-AUC 差异不显著。
5. SHAP 解释显示，不活跃时间、缺失评估、评估得分和评估完成情况是模型识别风险的重要依据。
6. 跨课程泛化和退课样本影响仍是需要谨慎讨论的限制。

## 12. 不能夸大的地方

论文中不建议写：

- “模型可以准确预测所有风险学生。”
- “XGBoost 显著优于所有模型。”
- “SHAP 证明了不活跃会导致学生挂科。”
- “完整课程窗口也是早期预警结果。”

更合适的写法是：

- “模型在 OULAD 数据集上表现出较好的早期风险识别能力。”
- “XGBoost、LightGBM 和 CatBoost 在第 56 天窗口表现接近。”
- “SHAP 结果表明模型主要依赖不活跃、缺失评估和评估表现等可解释信号。”
- “完整课程窗口作为性能上界，不作为早期预警结果。”

## 13. 主要文件位置

- 代码入口：src/run_experiments.py
- 重复实验与显著性检验：src/repeated_significance.py
- 泄漏审计：outputs/reports/leakage_audit.md
- 实验总结：outputs/reports/experiment_summary.md
- 中文实验报告：outputs/reports/paper_experiment_section.md
- 通俗项目报告：outputs/reports/plain_language_project_report.md
- 模型比较表：outputs/tables/model_comparison.csv
- 多窗口结果表：outputs/tables/time_window_results.csv
- 特征消融表：outputs/tables/feature_ablation_results.csv
- SHAP 特征表：outputs/tables/shap_top_features.csv
- 重复实验表：outputs/tables/repeated_seed_model_summary.csv
- 显著性检验表：outputs/tables/significance_tests.csv

## 14. 一句话总结

我们完成了一个可复现的 OULAD 在线学习风险预测实验：它能在课程早期基于学生背景、平台行为和评估表现识别潜在风险学生，并通过多窗口实验、消融实验、稳健性分析和 SHAP 解释说明模型什么时候有效、为什么有效，以及哪些结论需要谨慎表述。
