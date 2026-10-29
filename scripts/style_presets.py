# -*- coding: utf-8 -*-
"""论文图件画风预设：只改样式，不改任何数据。"""
PRESETS = {
    # A：学位论文·灰阶安全风（推荐）
    "thesis_a": dict(
        C=dict(primary="#24425F", blue="#5B7FA6", light="#EAF0F6", mid="#A9BED4",
               orange="#A65A2E", orange_l="#F6EDE6", green="#4F6B45", green_l="#EDF2EA",
               gray="#5A5A5A", lgray="#F0F0F0", red="#9E2A2B", gold="#8A6D3B",
               purple="#5B4A78", purple_l="#EFEAF5"),
        LW=1.1, BODY=10.5, TICK=9.5, SUBT=11.5, TITLE=13.0, BIG=14.5,
        axes_lw=0.8, line_lw=1.5, box_lw=0.9, round=1.0, dpi=300,
    ),
    # B：期刊风（Science/ Nature 风格：低饱和、细线、紧凑）
    "journal_b": dict(
        C=dict(primary="#0F4D92", blue="#3A6EA5", light="#E9F0F7", mid="#9CB8D4",
               orange="#B64342", orange_l="#F7ECEC", green="#42949E", green_l="#E8F2F3",
               gray="#4D4D4D", lgray="#F2F2F2", red="#B64342", gold="#D9A441",
               purple="#8274B3", purple_l="#EFEDF7"),
        LW=0.9, BODY=10.0, TICK=9.0, SUBT=11.0, TITLE=12.5, BIG=13.5,
        axes_lw=0.7, line_lw=1.2, box_lw=0.8, round=0.8, dpi=300,
    ),
}
