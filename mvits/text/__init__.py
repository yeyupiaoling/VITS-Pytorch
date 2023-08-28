from mvits.text import cleaners
from mvits.text.symbol import get_symbols


def text_to_sequence(text, symbols, cleaner_name):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
        cleaner_name: name of the cleaner functions to run the text through
        symbols: a list of phoneme characters
      Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = []
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    cleaner = getattr(cleaners, cleaner_name)
    if not cleaner:
        raise Exception('Unknown cleaner: %s' % cleaner_name)
    clean_text = cleaner(text)
    print(clean_text)
    print(f" length:{len(clean_text)}")
    for s in clean_text:
        if s not in symbol_to_id.keys():
            continue
        symbol_id = symbol_to_id[s]
        sequence += [symbol_id]
    print(f" length:{len(sequence)}")
    return sequence


def cleaned_text_to_sequence(cleaned_text, symbols):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        cleaned_text: string to convert to a sequence
        symbols: a list of phoneme characters
      Returns:
        List of integers corresponding to the symbols in the text
    """
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sequence = [symbol_to_id[s] for s in cleaned_text if s in symbol_to_id.keys()]
    return sequence


def clean_text_(text, cleaner_name):
    cleaner = getattr(cleaners, cleaner_name)
    if not cleaner:
        raise Exception('Unknown cleaner: %s' % cleaner_name)
    text = cleaner(text)
    return text
