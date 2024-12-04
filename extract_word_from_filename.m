function word = extract_word_from_filename(filename)
% Extract word from filename format: sp01a_w03_head.mp3
parts = split(filename, '_');
word = extractBefore(parts{3}, '.mp3');
end