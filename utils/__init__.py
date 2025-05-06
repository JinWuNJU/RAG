import unicodedata

def truncate_text_by_display_width(text, max_width):
    """
    根据显示宽度截取文本。
    
    :param text: 输入字符串
    :param max_width: 最大显示宽度（按英文字符计）
    :return: 截取后的字符串
    """
    truncated = []
    current_width = 0

    for char in text:
        # 判断字符的显示宽度：英文字符为1，中文等全角字符为2
        char_width = 2 if unicodedata.east_asian_width(char) in 'WF' else 1
        if current_width + char_width > max_width:
            break
        truncated.append(char)
        current_width += char_width

    return ''.join(truncated)