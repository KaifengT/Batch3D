

def get_windows_colorization_color():

    import ctypes
    from ctypes.wintypes import DWORD

    DwmGetColorizationColor = ctypes.windll.dwmapi.DwmGetColorizationColor
    DwmGetColorizationColor.argtypes = [ctypes.POINTER(DWORD), ctypes.POINTER(ctypes.c_bool)]
    DwmGetColorizationColor.restype = ctypes.HRESULT


    color = DWORD()
    opaque = ctypes.c_bool()
    result = DwmGetColorizationColor(ctypes.byref(color), ctypes.byref(opaque))
    if result == 0:  # S_OK
        return color.value
    else:
        return None

if __name__ == "__main__":
    color = get_windows_colorization_color()
    if color is not None:
        print(f"Windows colorization color is: #{color:06x}")
        print(color)
    else:
        print("Failed to get Windows colorization color")