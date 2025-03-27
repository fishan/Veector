# src/interface.py
import numpy as np
from datetime import datetime

def display_text(text, element_id=None):
    """Отображает текст."""
    element_id = element_id or f"text-{datetime.now().timestamp()}"
    return f"<div id='{element_id}'>{text}</div>"

def display_number(number, element_id=None):
    """Отображает число."""
    element_id = element_id or f"number-{datetime.now().timestamp()}"
    return f"<span id='{element_id}'>{number}</span>"

def display_image(image_data, element_id=None):
    """Отображает изображение."""
    element_id = element_id or f"image-{datetime.now().timestamp()}"
    # Предполагаем, что image_data - это base64 строка
    return f"<img id='{element_id}' src='data:image/png;base64,{image_data}'/>"

def create_button(text, onclick_action, element_id=None):
    """Создает кнопку."""
    element_id = element_id or f"button-{datetime.now().timestamp()}"
    return f"<button id='{element_id}' onclick='{onclick_action}'>{text}</button>"

def create_text_field(default_text="", element_id=None):
    """Создает текстовое поле."""
    element_id = element_id or f"text-field-{datetime.now().timestamp()}"
    return f"<input type='text' id='{element_id}' value='{default_text}'/>"

def create_slider(min_value, max_value, default_value, element_id=None):
    """Создает слайдер."""
    element_id = element_id or f"slider-{datetime.now().timestamp()}"
    return f"<input type='range' id='{element_id}' min='{min_value}' max='{max_value}' value='{default_value}'/>"

def create_grid_layout(elements, cols=3, element_id=None):
    """Создает сеточный макет."""
    element_id = element_id or f"grid-{datetime.now().timestamp()}"
    grid_html = f"<div id='{element_id}' style='display: grid; grid-template-columns: repeat({cols}, 1fr);'>"
    for element in elements:
        grid_html += f"<div>{element}</div>"
    grid_html += "</div>"
    return grid_html

def create_list_layout(elements, element_id=None):
    """Создает список элементов."""
    element_id = element_id or f"list-{datetime.now().timestamp()}"
    list_html = f"<ul id='{element_id}'>"
    for element in elements:
        list_html += f"<li>{element}</li>"
    list_html += "</ul>"
    return list_html

def create_tabbed_layout(tabs, element_id=None):
    """Создает макет с вкладками."""
    element_id = element_id or f"tabs-{datetime.now().timestamp()}"
    tab_headers = ""
    tab_contents = ""
    for i, (tab_name, tab_content) in enumerate(tabs.items()):
        tab_id = f"{element_id}-tab-{i}"
        tab_headers += f"<button onclick=\"showTab('{element_id}', '{tab_id}')\">{tab_name}</button>"
        tab_contents += f"<div id='{tab_id}' class='tab-content'> {tab_content}</div>"

    tabbed_html = f"""
    <div id='{element_id}' class='tab'>
        {tab_headers}
        {tab_contents}
    </div>
    <script>
    function showTab(elementId, tabId) {{
        var tabs = document.querySelectorAll('#' + elementId + ' .tab-content');
        tabs.forEach(function(tab) {{
            tab.style.display = 'none';
        }});
        document.getElementById(tabId).style.display = 'block';
    }}
    // Show the first tab by default
    document.addEventListener('DOMContentLoaded', function() {{
        var firstTab = document.querySelector('#' + elementId + ' .tab-content');
        if (firstTab) {{
            firstTab.style.display = 'block';
        }}
    }});
    </script>
    """
    return tabbed_html
# Example
def human_readable(tensor):
    if not isinstance(tensor, list) or len(tensor) < 4:
        return str(tensor)
    layer, coords, data, length = tensor[0]
    op = tensor[1][2]
    next_coords = tensor[4]
    return f"Layer: {layer}, Coords: {coords}, Data: {data}, Op: {op}, Next: {next_coords}"
