import os
import re
import sys

import opencc

dialects = {'SZ': 'suzhou_2', 'WX': 'wuxi_2', 'CZ': 'changzhou', 'HZ': 'hangzhou_2',
            'SX': 'shaoxing_2', 'NB': 'ningbo_2', 'JJ': 'jingjiang_2', 'YX': 'yixing_2',
            'JD': 'jiading_2', 'ZR': 'zhenru_2', 'PH': 'pinghu_2', 'TX': 'tongxiang_2',
            'JS': 'jiashan_2', 'HN': 'xiashi_2', 'LP': 'linping_2', 'XS': 'xiaoshan_2',
            'FY': 'fuyang_2', 'RA': 'ruao_2', 'CX': 'cixi_2', 'SM': 'sanmen_2',
            'TT': 'tiantai_2', 'WZ': 'wenzhou_2', 'SC': 'suichang_2', 'YB': 'youbu_2'}

converters = {}
ABS_PATH = os.path.dirname(os.path.realpath(__file__))

for dialect in dialects.values():
    try:
        converters[dialect] = opencc.OpenCC(os.path.join(ABS_PATH, 'chinese_dialect_lexicons', dialect))
    except:
        print(f'不支持方言：{dialect}', file=sys.stderr)


# 方言
def ngu_dialect_to_ipa(text, dialect):
    dialect = dialects[dialect]
    text = converters[dialect].convert(text).replace('-', '').replace('$', ' ')
    text = re.sub(r'[、；：]', '，', text)
    text = re.sub(r'\s*，\s*', ', ', text)
    text = re.sub(r'\s*。\s*', '. ', text)
    text = re.sub(r'\s*？\s*', '? ', text)
    text = re.sub(r'\s*！\s*', '! ', text)
    text = re.sub(r'\s*$', '', text)
    return text
