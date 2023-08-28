def get_symbols(cleaner_names):
    if cleaner_names == 'chinese_dialect_cleaners':
        _pad = '_'
        _punctuation = ',.!?~…─'
        _letters = '#Nabdefghijklmnoprstuvwxyzæçøŋœȵɐɑɒɓɔɕɗɘəɚɛɜɣɤɦɪɭɯɵɷɸɻɾɿʂʅʊʋʌʏʑʔʦʮʰʷˀː˥˦˧˨˩̥̩̃̚ᴀᴇ↑↓∅ⱼ '
        symbols = [_pad] + list(_punctuation) + list(_letters)
        return symbols
    elif cleaner_names == 'cjke_cleaners2':
        _pad = '_'
        _punctuation = ',.!?-~…'
        _letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '
        symbols = [_pad] + list(_punctuation) + list(_letters)
        return symbols
    else:
        raise Exception(f'不支持该方法：{cleaner_names}')
