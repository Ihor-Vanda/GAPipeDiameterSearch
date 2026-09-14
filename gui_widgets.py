"""Спільні елементи інтерфейсу: стиль таблиць, навігація по графіках, лог.

Ці помічники використовували обидва екрани — головне вікно і перегляд
результатів, — і жили методами класу `PipelineOptimizerApp`, хоча від
його стану не залежать: усе, що їм потрібно, передається аргументами.
Через це перегляд результатів неможливо було відділити: він викликав
`self.setup_treeview_style()`, `self.load_df_to_treeview()`,
`self.enable_mouse_navigation()`, `self.prevent_typing` і
`self._get_actual_diams()` — тобто був прив'язаний до застосунку п'ятьма
методами, жоден з яких насправді нічого від нього не хотів.

Тепер це функції модуля, і жоден із двох екранів не мусить знати про
інший. П'ятий метод, `_get_actual_diams`, сюди не переїхав: він читав
таблицю цін, тобто був не про інтерфейс, а про дані, і при цьому
дублював `ga_data.load_config` у слабшому вигляді. Його місце —
`ga_data.diameter_table` (§27.3).
"""

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from tkinter import ttk

#: Темна палітра таблиць — та сама, що в решті вікна.
TREE_BACKGROUND = "#2b2b2b"
TREE_FOREGROUND = "#ffffff"
TREE_SELECTED = "#1f538d"
TREE_HEADING = "#333333"
TREE_HEADING_ACTIVE = "#444444"
TREE_ROW_HEIGHT = 25
#: Крок масштабування колесом мишки.
ZOOM_STEP = 1.2


class GUIStream:
    """Перенаправляє `stdout` у текстове поле вікна і, за потреби, у файл."""

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.log_file = None

    def set_file(self, path):
        self.close()
        self.log_file = open(path, "a", encoding="utf-8")

    def write(self, message):
        self.text_widget.insert("end", message)
        self.text_widget.see("end")
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None


def prevent_typing(_event):
    """Зробити текстове поле тільки для читання, не гасячи прокрутку."""
    return "break"


def setup_treeview_style(widget):
    """Темний стиль для `ttk.Treeview`. `widget` потрібен лише як корінь стилю."""
    style = ttk.Style(widget)
    try:
        style.theme_use("default")
    except AttributeError:
        pass

    style.configure("Treeview", background=TREE_BACKGROUND, foreground=TREE_FOREGROUND,
                    rowheight=TREE_ROW_HEIGHT, fieldbackground=TREE_BACKGROUND,
                    borderwidth=0)
    style.map("Treeview", background=[("selected", TREE_SELECTED)])

    style.configure("Treeview.Heading", background=TREE_HEADING,
                    foreground=TREE_FOREGROUND, relief="flat",
                    font=("Arial", 11, "bold"))
    style.map("Treeview.Heading", background=[("active", TREE_HEADING_ACTIVE)])


def sort_treeview_column(tree, column, reverse):
    """Відсортувати таблицю за стовпцем — числами, якщо вони числа."""
    rows = [(tree.set(item, column), item) for item in tree.get_children("")]
    try:
        rows.sort(key=lambda pair: (float(pair[0]) if pair[0] and pair[0] != "nan"
                                    else -float("inf")), reverse=reverse)
    except ValueError:
        rows.sort(key=lambda pair: str(pair[0]).lower(), reverse=reverse)

    for index, (_, item) in enumerate(rows):
        tree.move(item, "", index)

    tree.heading(column, command=lambda: sort_treeview_column(tree, column, not reverse))


def load_dataframe(tree, frame):
    """Показати `DataFrame` у таблиці; заголовки стають кнопками сортування."""
    tree.delete(*tree.get_children())
    tree["columns"] = list(frame.columns)
    tree["show"] = "headings"

    for column in tree["columns"]:
        tree.heading(column, text=column,
                     command=lambda c=column: sort_treeview_column(tree, c, False))
        tree.column(column, width=100, anchor="center")

    for _, row in frame.iterrows():
        tree.insert("", "end", values=list(row))


def enable_mouse_navigation(fig, ax, canvas):
    """Панорамування мишкою і масштабування колесом для графіка."""
    toolbar = NavigationToolbar2Tk(canvas, canvas.get_tk_widget().master,
                                   pack_toolbar=False)
    toolbar.update()
    toolbar.pan()

    def zoom(event):
        if event.inaxes != ax:
            return
        scale = ZOOM_STEP if event.button == "up" else 1 / ZOOM_STEP
        x, y = event.xdata, event.ydata
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        ax.set_xlim([x - (x - xlim[0]) / scale, x + (xlim[1] - x) / scale])
        ax.set_ylim([y - (y - ylim[0]) / scale, y + (ylim[1] - y) / scale])
        canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", zoom)

