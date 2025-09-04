# modified from win32mica

import ctypes
import sys
import winreg

class MicaTheme:
    LIGHT = 0
    DARK = 1
    AUTO = 2
    
class MicaStyle:
    DEFAULT = 3
    ALT = 4

class DWM_SYSTEMBACKDROP_TYPE:
    DWMSBT_AUTO = 0
    DWMSBT_NONE = 1
    DWMSBT_MAINWINDOW = 2      # Mica
    DWMSBT_TRANSIENTWINDOW = 3 # Mica Alt
    DWMSBT_TABBEDWINDOW = 4

class AccentPolicy(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_uint),
        ("AccentFlags", ctypes.c_uint),
        ("GradientColor", ctypes.c_uint),
        ("AnimationId", ctypes.c_uint),
    ]

class WindowCompositionAttribute(ctypes.Structure):
    _fields_ = [
        ("Attribute", ctypes.c_int),
        ("Data", ctypes.POINTER(ctypes.c_int)),
        ("SizeOfData", ctypes.c_size_t),
    ]

class _MARGINS(ctypes.Structure):
    _fields_ = [
        ("cxLeftWidth", ctypes.c_int),
        ("cxRightWidth", ctypes.c_int),
        ("cyTopHeight", ctypes.c_int),
        ("cyBottomHeight", ctypes.c_int),
    ]


DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_SYSTEMBACKDROP_TYPE_EARLY = 1029 # 0x405
DWMWA_SYSTEMBACKDROP_TYPE = 38

def __read_registry(aKey, sKey, default, storage=winreg.HKEY_CURRENT_USER):
    registry = winreg.ConnectRegistry(None, storage)
    reg_keypath = aKey
    try:
        reg_key = winreg.OpenKey(registry, reg_keypath)
    except FileNotFoundError:
        return default
    except Exception as e:
        print(e)
        return default
    for i in range(1024):
        try:
            value_name, value, _ = winreg.EnumValue(reg_key, i)
            if value_name == sKey:
                return value
        except OSError:
            return default
        except Exception as e:
            print(e)
            return default


def ApplyMica(
        HWND: int,
        Theme:int = MicaTheme.LIGHT,
        Style:int = DWM_SYSTEMBACKDROP_TYPE.DWMSBT_TRANSIENTWINDOW,
    ) -> int:
    """Applies the mica backdrop effect on a specific hWnd

    Args:

        HWND(int):
            The handle to the window on which the effect has to be applied
        Theme(int):
            The theme of the backdrop effect: MicaTheme.DARK, MicaTheme.LIGHT, MicaTheme.AUTO
        Style(int):
            The style of the mica backdrop effect

    Returns:
        int: the integer result of the win32 api call to apply the mica backdrop effect. This value will equal to 0x32 if the system is not compatible with the mica backdrop
    """

    if HWND == 0:
        raise ValueError("The parameter HWND cannot be zero")
    if Theme not in (MicaTheme.DARK, MicaTheme.LIGHT, MicaTheme.AUTO):
        raise ValueError("The parameter ColorMode has an invalid value")

    
    try:
        try:
            HWND = int(HWND)
        except ValueError:
            HWND = int(str(HWND), 16)

        user32 = ctypes.windll.user32
        dwm = ctypes.windll.dwmapi

        SetWindowCompositionAttribute = user32.SetWindowCompositionAttribute
        DwmSetWindowAttribute = dwm.DwmSetWindowAttribute
        DwmExtendFrameIntoClientArea = dwm.DwmExtendFrameIntoClientArea
        
        if Theme == 1:
            themeToSet = 1
        elif Theme == 0:
            themeToSet = 0
        else:
            themeToSet = 0 if getSystemTheme() != 0 else 1

            
        DwmSetWindowAttribute(
            HWND,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(themeToSet)),
            ctypes.sizeof(ctypes.c_int),
        )

        if sys.platform == "win32" and sys.getwindowsversion().build >= 22000:
            Acp = AccentPolicy()
            Acp.GradientColor = int("00cccccc", base=16)
            Acp.AccentState = 5
            Acp.AccentPolicy = 19

            Wca = WindowCompositionAttribute()
            Wca.Attribute = 20
            Wca.SizeOfData = ctypes.sizeof(Acp)
            Wca.Data = ctypes.cast(ctypes.pointer(Acp), ctypes.POINTER(ctypes.c_int))

            Mrg = _MARGINS(-1, -1, -1, -1)

            o = DwmExtendFrameIntoClientArea(HWND, ctypes.byref(Mrg))
            try:
                o = SetWindowCompositionAttribute(HWND, Wca)
            except ctypes.ArgumentError:
                ...

            if sys.getwindowsversion().build < 22523:
                
                return DwmSetWindowAttribute(
                    HWND,
                    DWMWA_SYSTEMBACKDROP_TYPE_EARLY,
                    ctypes.byref(ctypes.c_int(Style)),
                    ctypes.sizeof(ctypes.c_int),
                )
            else:
                
                return DwmSetWindowAttribute(
                    HWND,
                    DWMWA_SYSTEMBACKDROP_TYPE,
                    ctypes.byref(ctypes.c_int(Style)),
                    ctypes.sizeof(ctypes.c_int),
                )
        else:
            print(
                f"Win32Mica Error: {sys.platform} version {sys.getwindowsversion().build} is not supported"
            )
            return 0x32
    except Exception as e:
        print("Win32mica: " + str(type(e)) + ": " + str(e))

def getSystemTheme() -> int:
    """ Get the current system theme """
    CurrentTheme = __read_registry(
        r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        "AppsUseLightTheme",
        0,
    )
    return CurrentTheme
