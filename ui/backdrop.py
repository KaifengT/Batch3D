import ctypes
import sys
import traceback
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication


DEFAULT_LIGHT_BG = (0.9, 0.9, 0.9, 1.0)
DEFAULT_DARK_BG = (0.1, 0.1, 0.1, 1.0)
TRANSPARENT_CSS = '#00000000'
TRANSLUCENT_BACKDROP_PLATFORMS = ('win32', 'darwin')


class MicaTheme:
    LIGHT = 0
    DARK = 1
    AUTO = 2


class DWM_SYSTEMBACKDROP_TYPE:
    DWMSBT_AUTO = 0
    DWMSBT_NONE = 1
    DWMSBT_MAINWINDOW = 2
    DWMSBT_TRANSIENTWINDOW = 3
    DWMSBT_TABBEDWINDOW = 4


DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_SYSTEMBACKDROP_TYPE_EARLY = 1029
DWMWA_SYSTEMBACKDROP_TYPE = 38


@dataclass
class BackdropPresentation:
    active: bool
    theme: str
    background_css: str
    fallback_color: Tuple[float, float, float, float]


def _current_platform(platform: Optional[str] = None) -> str:
    return platform or sys.platform


def has_system_backdrop(platform: Optional[str] = None) -> bool:
    return _current_platform(platform) in TRANSLUCENT_BACKDROP_PLATFORMS


def supports_backdrop_styles(platform: Optional[str] = None) -> bool:
    return _current_platform(platform) == 'win32'


def normalize_theme_name(theme) -> str:
    if theme is None:
        return 'auto'

    if isinstance(theme, int):
        if theme == MicaTheme.DARK:
            return 'dark'
        if theme == MicaTheme.LIGHT:
            return 'light'
        return 'auto'

    value = getattr(theme, 'value', theme)
    name = getattr(theme, 'name', None)
    text = str(name if isinstance(name, str) else value).strip().lower()
    if 'dark' in text:
        return 'dark'
    if 'light' in text:
        return 'light'
    if 'auto' in text:
        return 'auto'
    return text or 'auto'


def resolve_theme_name(theme, auto_is_dark=None) -> str:
    theme_name = normalize_theme_name(theme)
    if theme_name != 'auto':
        return theme_name

    if callable(auto_is_dark):
        return 'dark' if auto_is_dark() else 'light'
    if auto_is_dark is not None:
        return 'dark' if bool(auto_is_dark) else 'light'

    app = QApplication.instance()
    if app is not None:
        window_color = app.palette().color(QPalette.Window)
        return 'dark' if window_color.lightness() < 128 else 'light'

    return 'dark'


def rgba_to_css(color: Tuple[float, float, float, float]) -> str:
    r, g, b, a = color
    if a <= 0:
        return TRANSPARENT_CSS
    if a >= 1:
        return f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})'
    return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a:.3f})'


def fallback_background_color(theme='auto', auto_is_dark=None,
                              light_color=DEFAULT_LIGHT_BG,
                              dark_color=DEFAULT_DARK_BG) -> Tuple[float, float, float, float]:
    theme_name = resolve_theme_name(theme, auto_is_dark)
    return light_color if theme_name == 'light' else dark_color


def background_css(theme='auto', auto_is_dark=None, transparent=False,
                   light_color=DEFAULT_LIGHT_BG,
                   dark_color=DEFAULT_DARK_BG) -> str:
    if transparent:
        return TRANSPARENT_CSS
    return rgba_to_css(fallback_background_color(
        theme=theme,
        auto_is_dark=auto_is_dark,
        light_color=light_color,
        dark_color=dark_color,
    ))


def configure_translucent_window(widget, platform: Optional[str] = None):
    if widget is None or not has_system_backdrop(platform):
        return False

    try:
        widget.setAttribute(Qt.WA_TranslucentBackground, True)
        widget.setAutoFillBackground(False)
        return True
    except Exception:
        traceback.print_exc()
        return False


def set_widget_background(widget, css: str):
    if widget is None:
        return False

    try:
        widget.setStyleSheet(f'background-color: {css};')
        return True
    except Exception:
        traceback.print_exc()
        return False


def _theme_to_windows(theme):
    theme_name = normalize_theme_name(theme)
    if theme_name == 'dark':
        return MicaTheme.DARK
    if theme_name == 'light':
        return MicaTheme.LIGHT
    return MicaTheme.AUTO


def _win_id_from(widget_or_win_id):
    if hasattr(widget_or_win_id, 'winId'):
        return widget_or_win_id.winId()
    return widget_or_win_id


def _coerce_int_handle(value):
    try:
        return int(value)
    except Exception:
        return int(str(value), 0)


def _windows_system_theme() -> int:
    try:
        import winreg
    except Exception:
        return 0

    registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
    try:
        reg_key = winreg.OpenKey(
            registry,
            r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize',
        )
    except FileNotFoundError:
        return 0
    except Exception:
        return 0

    try:
        for i in range(1024):
            try:
                value_name, value, _ = winreg.EnumValue(reg_key, i)
                if value_name == 'AppsUseLightTheme':
                    return value
            except OSError:
                return 0
            except Exception:
                return 0
    finally:
        try:
            winreg.CloseKey(reg_key)
        except Exception:
            pass

    return 0


def _apply_windows_mica(HWND: int, theme: int = MicaTheme.LIGHT,
                         style: int = DWM_SYSTEMBACKDROP_TYPE.DWMSBT_TRANSIENTWINDOW) -> int:
    if HWND == 0:
        raise ValueError('The parameter HWND cannot be zero')
    if theme not in (MicaTheme.DARK, MicaTheme.LIGHT, MicaTheme.AUTO):
        raise ValueError('The parameter ColorMode has an invalid value')

    try:
        HWND = _coerce_int_handle(HWND)
        user32 = ctypes.windll.user32
        dwm = ctypes.windll.dwmapi

        SetWindowCompositionAttribute = user32.SetWindowCompositionAttribute
        DwmSetWindowAttribute = dwm.DwmSetWindowAttribute
        DwmExtendFrameIntoClientArea = dwm.DwmExtendFrameIntoClientArea

        if theme == MicaTheme.DARK:
            themeToSet = 1
        elif theme == MicaTheme.LIGHT:
            themeToSet = 0
        else:
            themeToSet = 0 if _windows_system_theme() != 0 else 1

        DwmSetWindowAttribute(
            HWND,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(themeToSet)),
            ctypes.sizeof(ctypes.c_int),
        )

        if sys.platform == 'win32' and sys.getwindowsversion().build >= 22000:
            class AccentPolicy(ctypes.Structure):
                _fields_ = [
                    ('AccentState', ctypes.c_uint),
                    ('AccentFlags', ctypes.c_uint),
                    ('GradientColor', ctypes.c_uint),
                    ('AnimationId', ctypes.c_uint),
                ]

            class WindowCompositionAttribute(ctypes.Structure):
                _fields_ = [
                    ('Attribute', ctypes.c_int),
                    ('Data', ctypes.POINTER(ctypes.c_int)),
                    ('SizeOfData', ctypes.c_size_t),
                ]

            class _MARGINS(ctypes.Structure):
                _fields_ = [
                    ('cxLeftWidth', ctypes.c_int),
                    ('cxRightWidth', ctypes.c_int),
                    ('cyTopHeight', ctypes.c_int),
                    ('cyBottomHeight', ctypes.c_int),
                ]

            Acp = AccentPolicy()
            Acp.GradientColor = int('00cccccc', base=16)
            Acp.AccentState = 5
            Acp.AccentPolicy = 19

            Wca = WindowCompositionAttribute()
            Wca.Attribute = 20
            Wca.SizeOfData = ctypes.sizeof(Acp)
            Wca.Data = ctypes.cast(ctypes.pointer(Acp), ctypes.POINTER(ctypes.c_int))

            Mrg = _MARGINS(-1, -1, -1, -1)

            DwmExtendFrameIntoClientArea(HWND, ctypes.byref(Mrg))
            try:
                SetWindowCompositionAttribute(HWND, Wca)
            except ctypes.ArgumentError:
                pass

            if sys.getwindowsversion().build < 22523:
                return DwmSetWindowAttribute(
                    HWND,
                    DWMWA_SYSTEMBACKDROP_TYPE_EARLY,
                    ctypes.byref(ctypes.c_int(style)),
                    ctypes.sizeof(ctypes.c_int),
                )

            return DwmSetWindowAttribute(
                HWND,
                DWMWA_SYSTEMBACKDROP_TYPE,
                ctypes.byref(ctypes.c_int(style)),
                ctypes.sizeof(ctypes.c_int),
            )

        print(
            f'Win32Mica Error: {sys.platform} version {sys.getwindowsversion().build} is not supported'
        )
        return 0x32
    except Exception as e:
        print('Win32mica: ' + str(type(e)) + ': ' + str(e))
        return 0x32


def _apply_windows_backdrop(widget_or_win_id, theme, style):
    try:
        win_id = _win_id_from(widget_or_win_id)
        win_id = _coerce_int_handle(win_id)
        if win_id == 0:
            return False
        result = _apply_windows_mica(win_id, _theme_to_windows(theme), style)
        return int(style) != int(DWM_SYSTEMBACKDROP_TYPE.DWMSBT_NONE) and result == 0
    except Exception:
        print(f'ApplyMica id {_win_id_from(widget_or_win_id)} failed')
        traceback.print_exc()
        return False


def _objc_object_from_id(view_id):
    import objc

    try:
        ptr = int(view_id)
    except Exception:
        try:
            ptr = int(str(view_id), 0)
        except Exception:
            ptr = objc.pyobjc_id(view_id)

    return objc.objc_object(c_void_p=ctypes.c_void_p(ptr))


def _nsview_from(widget_or_view_id):
    if hasattr(widget_or_view_id, 'winId'):
        view_id = widget_or_view_id.winId()
    else:
        view_id = widget_or_view_id
    return _objc_object_from_id(view_id)


def _call_if_available(obj, selector, *args):
    method = getattr(obj, selector, None)
    if callable(method):
        return method(*args)
    return None


def _push_backdrop_behind_content(host_view, content_view, backdrop_view):
    backdrop_layer = _call_if_available(backdrop_view, 'layer')
    content_layer = _call_if_available(content_view, 'layer')
    host_layer = _call_if_available(host_view, 'layer')

    if backdrop_layer is not None:
        _call_if_available(backdrop_layer, 'setZPosition_', -1000.0)
        _call_if_available(backdrop_layer, 'setOpaque_', False)

    if host_layer is not None and content_layer is not None and backdrop_layer is not None:
        try:
            host_layer.insertSublayer_below_(backdrop_layer, content_layer)
        except Exception:
            pass


def _remove_old_backdrop(widget_or_view_id):
    old_view = getattr(widget_or_view_id, '_b3d_backdrop_view', None)
    if old_view is not None:
        try:
            old_view.removeFromSuperview()
        except Exception:
            pass
        try:
            widget_or_view_id._b3d_backdrop_view = None
        except Exception:
            pass


def _macos_tint_color(AppKit, theme):
    theme_name = resolve_theme_name(theme)
    if theme_name == 'light':
        return AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.24)
    if theme_name == 'dark':
        return AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.18)
    return AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.16)


def _apply_macos_backdrop(widget_or_view_id, theme, style):
    if int(style) == int(DWM_SYSTEMBACKDROP_TYPE.DWMSBT_NONE):
        _remove_old_backdrop(widget_or_view_id)
        return False

    try:
        import AppKit
        import objc  # noqa: F401
    except Exception:
        return False

    try:
        ns_view = _nsview_from(widget_or_view_id)
        ns_window = ns_view.window()
        if ns_window is None:
            return False

        content_view = ns_window.contentView()
        if content_view is None:
            return False
        host_view = content_view.superview() or content_view

        ns_window.setOpaque_(False)
        ns_window.setBackgroundColor_(AppKit.NSColor.clearColor())

        _remove_old_backdrop(widget_or_view_id)

        frame = host_view.bounds()
        glass_view_class = getattr(AppKit, 'NSGlassEffectView', None)
        if glass_view_class is not None:
            backdrop_view = glass_view_class.alloc().initWithFrame_(frame)
            backdrop_view.setStyle_(
                getattr(AppKit, 'NSGlassEffectViewStyleRegular', 1)
            )
            tint = _macos_tint_color(AppKit, theme)
            _call_if_available(backdrop_view, 'setTintColor_', tint)
        else:
            backdrop_view = AppKit.NSVisualEffectView.alloc().initWithFrame_(frame)
            material = getattr(
                AppKit,
                'NSVisualEffectMaterialHUDWindow',
                getattr(AppKit, 'NSVisualEffectMaterialUnderWindowBackground', 21),
            )
            backdrop_view.setMaterial_(material)
            backdrop_view.setBlendingMode_(
                getattr(AppKit, 'NSVisualEffectBlendingModeBehindWindow', 0)
            )
            backdrop_view.setState_(getattr(AppKit, 'NSVisualEffectStateActive', 1))

        backdrop_view.setAutoresizingMask_(
            getattr(AppKit, 'NSViewWidthSizable', 2) |
            getattr(AppKit, 'NSViewHeightSizable', 16)
        )
        _call_if_available(backdrop_view, 'setWantsLayer_', True)
        _call_if_available(backdrop_view, 'setIgnoresMouseEvents_', True)

        if host_view is not content_view:
            host_view.addSubview_positioned_relativeTo_(
                backdrop_view,
                getattr(AppKit, 'NSWindowBelow', -1),
                content_view,
            )
        else:
            content_view.addSubview_(backdrop_view)

        _push_backdrop_behind_content(host_view, content_view, backdrop_view)

        if hasattr(widget_or_view_id, 'setAttribute'):
            configure_translucent_window(widget_or_view_id, platform='darwin')
            widget_or_view_id._b3d_backdrop_view = backdrop_view

        return True
    except Exception:
        print('macOS system backdrop unavailable; using solid Qt background.')
        traceback.print_exc()
        return False


def clear_system_backdrop(widget_or_view_id, platform: Optional[str] = None):
    platform = _current_platform(platform)
    if platform == 'win32':
        _apply_windows_backdrop(
            widget_or_view_id,
            'auto',
            DWM_SYSTEMBACKDROP_TYPE.DWMSBT_NONE,
        )
        return True
    if platform == 'darwin':
        _remove_old_backdrop(widget_or_view_id)
        return True
    return False


def apply_system_backdrop(widget_or_win_id, theme='auto',
                          style=DWM_SYSTEMBACKDROP_TYPE.DWMSBT_TRANSIENTWINDOW,
                          platform: Optional[str] = None):
    platform = _current_platform(platform)
    if style is None:
        style = DWM_SYSTEMBACKDROP_TYPE.DWMSBT_TRANSIENTWINDOW

    if platform == 'win32':
        return _apply_windows_backdrop(widget_or_win_id, theme, style)
    if platform == 'darwin':
        return _apply_macos_backdrop(widget_or_win_id, theme, style)
    return False


class BackdropController:
    def __init__(self, theme='auto',
                 style=DWM_SYSTEMBACKDROP_TYPE.DWMSBT_TRANSIENTWINDOW,
                 auto_is_dark=None,
                 platform: Optional[str] = None,
                 light_color=DEFAULT_LIGHT_BG,
                 dark_color=DEFAULT_DARK_BG) -> None:
        self.platform = _current_platform(platform)
        self.theme = theme
        self.style = style
        self.auto_is_dark = auto_is_dark
        self.light_color = light_color
        self.dark_color = dark_color
        self._active = False
        self.windows = []
        self.background_widgets = []
        self.opengl_widgets = []

    @property
    def active(self) -> bool:
        return self._active

    def bind(self, windows: Optional[Iterable] = None,
             background_widgets: Optional[Iterable] = None,
             opengl_widgets: Optional[Iterable] = None):
        if windows is not None:
            self.windows = [w for w in windows if w is not None]
        if background_widgets is not None:
            self.background_widgets = [w for w in background_widgets if w is not None]
        if opengl_widgets is not None:
            self.opengl_widgets = [w for w in opengl_widgets if w is not None]
        return self

    def set_theme(self, theme):
        self.theme = theme
        return self

    def set_style(self, style):
        self.style = style
        return self

    def effective_theme_name(self) -> str:
        return resolve_theme_name(self.theme, self.auto_is_dark)

    def effective_theme(self):
        return 'dark' if self.effective_theme_name() == 'dark' else 'light'

    def fallback_color(self):
        return fallback_background_color(
            theme=self.theme,
            auto_is_dark=self.auto_is_dark,
            light_color=self.light_color,
            dark_color=self.dark_color,
        )

    def prefers_transparent_background(self) -> bool:
        return has_system_backdrop(self.platform) and int(self.style) != int(
            DWM_SYSTEMBACKDROP_TYPE.DWMSBT_NONE
        )

    def background_css(self, theme=None, transparent=None) -> str:
        if theme is None:
            theme = self.theme
        if transparent is None:
            transparent = self._active
        return background_css(
            theme=theme,
            auto_is_dark=self.auto_is_dark,
            transparent=transparent,
            light_color=self.light_color,
            dark_color=self.dark_color,
        )

    def apply_window(self, widget_or_win_id, style=None) -> bool:
        if style is None:
            style = self.style
        return apply_system_backdrop(
            widget_or_win_id,
            theme=self.theme,
            style=style,
            platform=self.platform,
        )

    def style_window(self, widget_or_win_id, theme=None, style=None) -> bool:
        if theme is None:
            theme = self.theme
        if style is None:
            style = self.style
        applied = self.apply_window(widget_or_win_id, style=style)
        set_widget_background(
            widget_or_win_id,
            self.background_css(theme=theme, transparent=applied),
        )
        return applied

    def sync(self, windows: Optional[Iterable] = None,
             background_widgets: Optional[Iterable] = None,
             opengl_widgets: Optional[Iterable] = None):
        if windows is None:
            windows = self.windows
        if background_widgets is None:
            background_widgets = self.background_widgets
        if opengl_widgets is None:
            opengl_widgets = self.opengl_widgets

        applied = False
        for window in windows:
            configure_translucent_window(window, platform=self.platform)
            applied = self.apply_window(window) or applied

        self._active = applied

        css = self.background_css(transparent=applied)
        for widget in background_widgets:
            set_widget_background(widget, css)

        fallback = self.fallback_color()
        for widget in opengl_widgets:
            setter = getattr(widget, 'setSystemBackdropEnabled', None)
            if callable(setter):
                setter(applied, fallback)

        return applied

    def schedule_sync(self, delay_ms: int = 0,
                      windows: Optional[Iterable] = None,
                      background_widgets: Optional[Iterable] = None,
                      opengl_widgets: Optional[Iterable] = None):
        QTimer.singleShot(
            delay_ms,
            lambda: self.sync(
                windows=windows,
                background_widgets=background_widgets,
                opengl_widgets=opengl_widgets,
            ),
        )
