import re

# Read file
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# More robust: use emoji range pattern
# Match high and low surrogates, and specific emoji chars
def strip_emoji(text):
    # Wide Unicode emojis
    result = []
    for ch in text:
        code = ord(ch)
        # Skip if it's an emoji/symbol block
        if 0x1F300 <= code <= 0x1FAFF:  # Emoticons + symbols + extended
            continue
        if 0x2600 <= code <= 0x27BF:  # Misc symbols + dingbats
            continue
        if 0x2300 <= code <= 0x23FF:  # Technical symbols
            continue
        if 0x2B00 <= code <= 0x2BFF:  # Arrows
            continue
        if 0x25A0 <= code <= 0x25FF:  # Geometric shapes
            continue
        if 0x1F100 <= code <= 0x1F1FF:  # Flags
            continue
        if 0x1F200 <= code <= 0x1F2FF:  # Enclosed chars
            continue
        if 0xFE00 <= code <= 0xFE0F:  # Variation selectors
            continue
        if 0x2100 <= code <= 0x214F:  # Letterlike
            continue
        if 0x2190 <= code <= 0x21FF:  # Arrows
            continue
        if 0x2000 <= code <= 0x206F:  # General punct
            # Keep regular punctuation
            if code in [0x2018, 0x2019, 0x201C, 0x201D, 0x2010, 0x2013, 0x2014]:
                result.append(ch)
            continue
        result.append(ch)
    return ''.join(result)

clean = strip_emoji(content)

# Remove specific known problematic sequences
specific = [
    '🤖', '👤', '🔎', '📌', '📊', '🧩', '🎯', '🧼', '✅', '⚙️', 
    '🏆', '⚖️', '📋', '💡', '📝', '🚀', '🎨', '🖼️', '📈', '📉', 
    '🔧', '🔨', '⚡', '🔥', '✨', '💯', '🌐', '💻', '🖥️', '📱', 
    '🧠', '⚠️', '❌', '⭕', '👁️', '🗣️', '🎵', '🎶', '➕', '➖', 
    '➗', '✖️', '♾️', '💲', '💱', '™️', '©️', '®️', '❗', '❕', 
    '❓', '❔', '‼️', '⁉️', '🔅', '🔆', '〽️', '🚸', '🔱', '⚜️', 
    '🔰', '🛜', '❇️', '✳️', '❎', '💠', 'Ⓜ️', '🌀', '💤', '🏧', 
    '🚾', '♿', '🅿️', '🈳', '🈂️', '🛂', '🛃', '🛄', '🛅', '🚹', 
    '🚺', '🚼', '🚻', '🚮', '🎦', '📶', '🈁', '🔣', 'ℹ️', '🔤', 
    '🔡', '🔠', '🆖', '🆗', '🆙', '🆒', '🆕', '🆓', '▶️', '⏸️', 
    '⏯️', '⏹️', '⏺️', '⏭️', '⏮️', '⏩', '⏪', '⏫', '⏬', '◀️', 
    '🔼', '🔽', '➡️', '⬅️', '⬆️', '⬇️', '↗️', '↘️', '↙️', '↖️', 
    '↕️', '↔️', '↪️', '↩️', '⤴️', '⤵️', '🔀', '🔁', '🔂', '🔄', 
    '🔃', '💉', '🩸', '💊', '🩹', '🩺', '🩻', '🧬', '🧪', '🧫', 
    '🌡️', '🧹', '🧺', '🧻', '🧼', '🪒', '🧽', '🧴', '🛁', '🚿', 
    '🛝', '🪥', '🧵', '🪡', '🧶', '🧰', '🧲', '⚖️', '🔗', '⛓️', 
    '🧷', '🔩', '🔨', '⛏️', '⚒️', '🛠️', '🗜️', '🪛', '🔧', '🎖️',
    '0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣',
    '🔟', '🔢', '#️⃣', '*️⃣', '💮', '🉐', '㊙️', '㊗️', '🈴', '🈵',
    '🈹', '🈲', '🅰️', '🅱️', '🆎', '🆑', '🅾️', '🆘', '🛑', '⛔',
    '📛', '🚫', '💢', '♨️', '🚷', '🚯', '🚳', '🚱', '🔞', '📵',
    '🚭', '🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘',
    '🕙', '🕚', '🕛', '🕜', '🕝', '🕞', '🕟', '🕠', '🕡', '🕢',
    '🕣', '🕤', '🕥', '🕦', '🕧', '🛎️', '🔔', '🔕', '⏰', '⏲️',
    '⏱️', '🕰️', '⏳', '⌛', '📣', '📢', '📯', '🔇', '🔈', '🔉',
    '🔊', '🎼', '🎹', '🥁', '🪘', '🎷', '🎺', '🪗', '🎸', '🪕',
    '🎻', '🪈', '🎬', '🎞️', '🎥', '📽️', '📺', '📷', '📸', '📹',
    '📼', '🔍', '🔬', '🔭', '📡', '💡', '🔦', '🏮', '🪔', '📔',
    '📕', '📖', '📗', '📘', '📙', '📚', '📓', '📒', '📃', '📜',
    '📄', '📰', '🗞️', '📑', '🔖', '🏷️', '💰', '💴', '💵', '💶',
    '💷', '💸', '💳', '🧾', '💹', '✉️', '📧', '📨', '📩', '📤',
    '📥', '📦', '📫', '📪', '📬', '📭', '📮', '🗳️', '✏️', '✒️',
    '🖋️', '🖊️', '🖌️', '🖍️', '📝', '💼', '📁', '📂', '🗂️', '📅',
    '📆', '🗒️', '🗓️', '📇', '📋', '📍', '📎', '🖇️', '📏', '📐',
    '✂️', '🗃️', '🗄️', '🗑️', '🔒', '🔓', '🔏', '🔐', '🔑', '🗝️',
    '🔨', '⛏️', '⚒️', '🛠️', '🗡️', '💣', '🪓', '🔪', '🏹', '🛡️',
]

for e in specific:
    clean = clean.replace(e, '')

# Also replace empty icon string and avatar defaults
clean = clean.replace("page_icon=''", "page_icon='🧠'")  # No wait, we need no emoji. Actually st.set_page_config can take page_icon as None or a path
clean = clean.replace("page_icon=''", "page_icon=None")
# Remove empty left from replacements
clean = clean.replace("  ", " ")
clean = clean.replace("\n\n\n", "\n\n")

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(clean)

print('Done - emojis removed from streamlit_app.py')
