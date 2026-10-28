# v1.1 可复现源文件说明

## 环境
- Python 3.12（容器镜像 local-ai:latest，含 python-docx 1.2.0）
- matplotlib 3.10.1 + numpy 2.5.4（工作区 pylibs，pip --target 安装）
- 中文字体：SimHei（fonts/simhei.ttf）

## 重新生成论文
```bash
python build_v11.py
# 输出：07_姚烨_2024245834_Z公司工作流程数字化再造研究_v1.1_核验修订版.docx
```

## 重新生成图件（可选）
```bash
python make_figures_v11.py   # 图1-1/2-1/5-2/6-1/6-2/6-3，输出到 figures_v11/
python make_figures_v12.py   # 图6-4，输出到 figures_final/
```

## 改文字
- 第1-2章：text_part1.py
- 第3-4章：text_part2.py
- 第5-7章 + 附录：text_part3.py
- 表格数据：build_v11.py 的 TABLES 字典（含表6-3 试点期实测基线）
- 参考文献：refs_v11.py（52 条，顺序编码制，编号=正文首次引用顺序）

## 注意
- 参考文献编号必须与正文首次引用顺序一致；增删文献后需同步检查正文 [n]。
- 图件文件名与 text_part*.py 中 ("fig", 文件名, ...) 元组必须一致。
- 目录为 Word 自动域，打开后右键"更新域"生成页码。
- 表6-3 与图6-4 使用作者提供的经许可聚合数据（四舍五入处理），引用时保留"经聚合与四舍五入处理"的说明。
