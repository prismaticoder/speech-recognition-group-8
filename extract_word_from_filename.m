function word = extract_word_from_filename(filename)
% Extract last word before .mp3 extension from filename (e.g., 'head' from 'sp01a_w03_head.mp3')
parts = split(filename, '_');
word = extractBefore(parts{end}, '.mp3');
end