# Quick fix for Gesi data loading
# This script will add the elif block for Gesi in handle_map_selection_v2

import re

file_path = r"c:\Users\lenovo\Desktop\Yeni_klasör\projects\AFDGCN_Garnoldi\gradio_app_enhanced.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the function
old_pattern = r'''(def handle_map_selection_v2\(kavsak, current_data_type\):
    choices = ANALYSIS_INTERSECTIONS\.get\(kavsak, \[\]\)
    if kavsak in KIZILIRMAK_LIST:
        dates = DATES_KIZILIRMAK
        return \(
            gr\.update\(choices=choices, value=choices\[0\] if choices else None\),  # c_in
            gr\.update\(choices=dates, value=dates\[0\] if dates else None\),        # t_in
            gr\.update\(visible=False\)                                            # data_type_in
        \)
    else:)'''

new_code = r'''def handle_map_selection_v2(kavsak, current_data_type):
    choices = ANALYSIS_INTERSECTIONS.get(kavsak, [])
    if kavsak in KIZILIRMAK_LIST:
        dates = DATES_KIZILIRMAK
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),  # c_in
            gr.update(choices=dates, value=dates[0] if choices else None),        # t_in
            gr.update(visible=False)                                            # data_type_in
        )
    elif kavsak == "Gesi":
        # Gesi kasım verisi kullanıyor, otomatik olarak seç
        dates = DATES_ILDEM.get("normal veri(kasım)", [])
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),  # c_in
            gr.update(choices=dates, value=dates[0] if choices else None),        # t_in
            gr.update(visible=True, value="normal veri(kasım)")                 # data_type_in
        )
    else:'''

content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Gesi handling added to handle_map_selection_v2")
