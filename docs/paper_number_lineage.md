# 数字谱系（Paper Number Lineage）

> 手稿中每一个实验数字的出处登记。格式：数字 → 生成脚本 → 运行产物（JSON manifest）→ 入稿位置。
> 规则：数字未登记不得入稿；重跑后必须更新本表与手稿。

## 5.3 构念验证（2026-10-23 入稿，gate 通过）

脚本：`src/validate/b_construct.py`（seed=42，n_perm=1000，n_boot=1000，
每家族配对上限 100，组间样本 = 3×组内）。产物：`runs/b/*.json`。

| 数字 | 出处 |
|---|---|
| B1 contrast 0.240 / 0.155 / 0.118（a1/a2/a3） | `runs/b/mtc.a{1,2,3}.json` → b1.contrast |
| B1 CI [0.231,0.248] / [0.147,0.163] / [0.112,0.125] | 同上 → b1.ci95（家族级 cluster bootstrap） |
| B1 p=0.001（1000 次置换无一达到观测值） | 同上 → b1.p_perm |
| B1 样本量 11,960 片段 / 1,979 家族 / 22,972 vs 68,916 对 | 同上 → b1.n_* |
| B2 r=0.44 / 0.45 / 0.43，p=0.001，91,888 对 | `runs/b/mtc.a{1,2,3}.json` → b2.mantel_r / p_perm / n_pairs |
| B3 rho=0.016/0.043/0.061，p=0.018/0.001/0.001 | `runs/b/pop909.a{1,2,3}.json` → b3.spearman_rho / p_perm |
| B3 四分位对比 0.001 / 0.020 / 0.017 | 同上 → b3.quartile_contrast |
| B3 样本量 908 首 / 18,160 对 | 同上 → b3.n_pieces / n_pairs |

设计修正（已入稿声明）：POP909 公开发布无年代元数据，原 B3"年代结构"改为
"和声词汇一致性"检验；S3 的"按专辑分组"表述改为"由和弦标注构成和声词汇分组"。
B3 在 a1 编码下效应可忽略（rho=0.016，四分位差 0.001）——手稿如实报告为
"supported but weaker"。

## 图表产物登记

| 图表 | 内容 | 生成路径 | 入稿位置 |
|---|---|---|---|
| Table T2 | B1/B2/B3 九项统计量汇总（三编码×三检验） | 手动登记于 `runs/b/*.json` | 5.3 Results 段后（三线表） |
| Figure F4 | MTC 组内/组间轮廓 Jaccard ECDF 三联图 | `figures/figF4_ecdf.svg` → Inkscape → `figures/figF4_ecdf.emf`（嵌入 `word/media/image1.emf`） | 5.3 Table TV 之后 |

图数据：`runs/b/jac.mtc.a{1,2,3}.npz`（b_construct.py 加 dump_path 后重跑，
统计量逐位复现，谱系不变）。
