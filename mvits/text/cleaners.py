import re

from mvits.text.english import english_to_ipa2, english_to_lazy_ipa2
from mvits.text.japanese import japanese_to_ipa2, japanese_to_ipa3
from mvits.text.korean import korean_to_ipa
from mvits.text.mandarin import chinese_to_ipa, chinese_to_ipa2


# noinspection RegExpRedundantEscape
# 中日韩英
def cjke_cleaners2(text):
    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: chinese_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: english_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


# noinspection RegExpRedundantEscape
# 中日英+方言
def chinese_dialect_cleaners(text):
    from mvits.text.cantonese import cantonese_to_ipa
    from mvits.text.shanghainese import shanghainese_to_ipa
    from mvits.text.ngu_dialect import ngu_dialect_to_ipa
    # 中文
    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: chinese_to_ipa2(x.group(1)) + ' ', text)
    # 日语
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_ipa3(x.group(1)).replace('Q', 'ʔ') + ' ', text)
    # 英语
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: english_to_lazy_ipa2(x.group(1)) + ' ', text)
    # 粤语
    text = re.sub(r'\[GD\](.*?)\[GD\]', lambda x: cantonese_to_ipa(x.group(1)) + ' ', text)
    # 上海话
    text = re.sub(r'\[SH\](.*?)\[SH\]', lambda x: shanghainese_to_ipa(x.group(1)).replace('1', '˥˧')
                  .replace('5', '˧˧˦').replace('6', '˩˩˧').replace('7', '˥').replace('8', '˩˨').replace('ᴀ', 'ɐ')
                  .replace('ᴇ', 'e') + ' ', text)
    # 其他方言
    text = re.sub(r'\[([A-Z]{2})\](.*?)\[\1\]', lambda x: ngu_dialect_to_ipa(x.group(2), x.group(1))
                  .replace('ʣ', 'dz').replace('ʥ', 'dʑ').replace('ʦ', 'ts').replace('ʨ', 'tɕ') + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text
