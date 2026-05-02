# 有效性与数据泄漏控制

## 1. 总体说明

在线学习风险预测实验容易受到数据泄漏影响，尤其是在使用课程后期行为、未来评估成绩、退课日期或学生标识符时。为降低这些风险，本实验在特征构建、数据划分、预处理和结果报告阶段均设置了显式控制。相关审计结果已由代码生成并保存于 outputs/reports/leakage_audit.md。

审计报告显示，当前流水线在主要泄漏风险项上均为 PASS，包括未来 VLE 数据、未来评估成绩、`final_result` 作为特征、`id_student` 作为特征、退课日期泄漏、预处理在全数据上拟合、类别不平衡处理和结果文件可复现性等（outputs/reports/leakage_audit.md）。

## 2. 目标变量与禁止特征

实验标签由 `final_result` 转换得到，其中 Fail 和 Withdrawn 被定义为风险类，Pass 和 Distinction 被定义为非风险类。为避免直接标签泄漏，`final_result` 仅用于构造目标变量，不进入模型特征。审计报告显示，各预测窗口的模型特征集中均不存在目标字段、学生标识字段、课程标识字段或退课字段（outputs/reports/leakage_audit.md）。

`id_student` 仅用于数据表连接、训练-测试划分分组以及局部 SHAP 解释标识，不作为模型输入特征。`date_unregistration` 具有直接揭示退课状态的风险，因此不被合并进建模特征。审计报告中 “final_result as a feature”“id_student as a feature” 和 “Withdrawal date leakage” 均为 PASS（outputs/reports/leakage_audit.md）。

## 3. 时间窗口泄漏控制

多窗口早期预测要求每个预测窗口只能使用预测日及其之前可观测到的数据。VLE 特征构建时仅保留 `date <= prediction_day` 的交互记录。审计报告中各窗口的 “VLE activity occurs on/before prediction day” 均为 PASS，且 violations = 0（outputs/reports/leakage_audit.md）。

评估特征构建时同时要求评估截止日期和学生提交日期均不晚于预测日。审计报告中各窗口的 “assessment submissions occur on/before prediction day” 均为 PASS，且 violations = 0（outputs/reports/leakage_audit.md）。因此，模型不会在第 7 天、第 14 天、第 28 天或第 56 天窗口中使用未来提交的评估成绩。

注册时间特征也进行了窗口化处理。若学生注册日期晚于预测日，则 `days_before_start_registered` 被置为缺失，而不是使用未来注册日期。审计报告显示，第 7 天、第 14 天、第 28 天、第 56 天和完整课程窗口中均无违反该规则的记录；对应的 future-or-missing registration rows 分别为 138、103、61、54 和 45，violations 均为 0（outputs/reports/leakage_audit.md）。

## 4. 训练-测试划分有效性

主验证协议采用按学生标识分组的分层训练-测试划分，避免同一学生的多个课程记录同时出现在训练集和测试集中。审计报告中的 Split Audit 显示，五个预测窗口的训练集均包含 26,074 行和 23,029 名唯一学生，测试集均包含 6,519 行和 5,756 名唯一学生，训练集与测试集的 overlapping_students 均为 0（outputs/reports/leakage_audit.md）。

此外，实验还报告了课程开课学期层面的泛化验证。GroupKFold by course presentation 使用 5 折，leave-one-course-presentation-out 使用 22 折。两种协议的结果均保存于 outputs/tables/model_comparison.csv。它们用于补充说明模型在跨课程开课学期场景下的泛化表现。

## 5. 预处理与类别不平衡控制

所有模型均通过 Pipeline 训练，预处理器与分类器在每个训练划分或训练折内部一起拟合。数值特征填补、标准化、类别特征填补和 one-hot 编码均不在完整数据集上预先拟合。审计报告中 “Preprocessing fitted on full dataset” 为 PASS（outputs/reports/leakage_audit.md）。

类别不平衡未通过划分前重采样处理。实验使用模型内部权重机制，包括 class weight、balanced subsample 和由训练集标签计算的 `scale_pos_weight`。审计报告中 “Class imbalance handling” 为 PASS，并说明未使用 SMOTE 或全数据重采样（outputs/reports/leakage_audit.md）。

## 6. 结果可复现性

所有主要表格和图像均由 `src/run_experiments.py` 从原始 CSV 文件生成。结果文件包括 outputs/tables/dataset_summary.csv、outputs/tables/model_comparison.csv、outputs/tables/time_window_results.csv、outputs/tables/feature_ablation_results.csv、outputs/tables/shap_top_features.csv，以及对应的图像文件。审计报告中 “Non-reproducible figures or tables” 为 PASS，并确认所有必需的表格、图像和报告均存在于 outputs/ 目录下（outputs/reports/leakage_audit.md）。

## 7. 对内部有效性的影响

上述控制降低了几类主要内部有效性威胁。第一，未来行为与未来评估成绩被排除在早期预测窗口之外。第二，标签字段和学生标识符未作为模型输入。第三，主训练-测试划分避免了重复学生跨集合出现。第四，预处理参数未在测试集上拟合。第五，类别不平衡处理未在数据划分前执行。

因此，当前结果可以作为该实验设置下的有效执行结果进行报告。不过，这些控制不意味着模型解释具有因果含义。SHAP 结果反映模型在给定训练数据和特征表示下的关联性贡献，而不能证明学习行为对最终学业结果的因果影响。

## 8. 剩余局限

当前实验仍存在若干有效性限制。首先，完整课程窗口使用课程全周期内的交互和评估信息，因此只能作为性能上界，不应作为早期预警场景解释。其次，尽管主划分是 student-disjoint，课程开课学期分组验证并不保证同一学生不会在不同课程开课学期之间跨折出现。第三，样本中仍包含可能在预测日前已经退课的学生；虽然 `date_unregistration` 未被使用，但此类学生的后续不活跃行为可能影响风险预测，应在后续敏感性实验中进一步排除或单独分析。第四，当前结果主要报告判别指标，尚未包含校准分析、置信区间或决策阈值效用分析（outputs/reports/leakage_audit.md）。

