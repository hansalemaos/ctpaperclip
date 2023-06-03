import ctypes
import io
import os.path
import sys
from collections import defaultdict, deque
from ctypes.wintypes import HANDLE, UINT, LPVOID, BOOL, HWND
from functools import reduce
from pprint import pprint
import kthread
from ctypes_window_info import get_window_infos
from deepcopyall import deepcopy
import numpy as np
from io import BytesIO
from PIL import Image, BmpImagePlugin
from kthread_sleep import sleep
from mousekey import MouseKey
from tolerant_isinstance import isinstance_tolerant

mkey = MouseKey()
windll = ctypes.LibraryLoader(ctypes.WinDLL)
GHND = 0x0042
GMEM_SHARE = 0x2000
CF_BITMAP = 2
CF_DIB = 8
CF_DIBV5 = 17
CF_DIF = 5
CF_DSPBITMAP = 130
CF_DSPENHMETAFILE = 142
CF_DSPMETAFILEPICT = 131
CF_DSPTEXT = 129
CF_ENHMETAFILE = 14
CF_GDIOBJFIRST = 768
CF_GDIOBJLAST = 1023
CF_HDROP = 15
CF_LOCALE = 16
CF_METAFILEPICT = 3
CF_OEMTEXT = 7
CF_OWNERDISPLAY = 128
CF_PALETTE = 9
CF_PENDATA = 10
CF_PRIVATEFIRST = 512
CF_PRIVATELAST = 767
CF_RIFF = 11
CF_SYLK = 4
CF_TEXT = 1
CF_TIFF = 6
CF_UNICODETEXT = 13
CF_WAVE = 12
GlobalAlloc = windll.kernel32.GlobalAlloc
GlobalAlloc.restype = HANDLE
GlobalAlloc.argtypes = [UINT, ctypes.c_size_t]

GlobalLock = windll.kernel32.GlobalLock
GlobalLock.restype = LPVOID
GlobalLock.argtypes = [HANDLE]

GlobalUnlock = windll.kernel32.GlobalUnlock
GlobalUnlock.restype = BOOL
GlobalUnlock.argtypes = [HANDLE]

HeapLock = windll.kernel32.HeapLock
HeapLock.restype = BOOL
HeapLock.argtypes = [HANDLE]

HeapUnlock = windll.kernel32.HeapUnlock
HeapUnlock.restype = BOOL
HeapUnlock.argtypes = [HANDLE]

OpenClipboard = windll.user32.OpenClipboard
OpenClipboard.restype = BOOL
OpenClipboard.argtypes = [HWND]

GlobalSize = windll.kernel32.GlobalSize
GlobalSize.restype = ctypes.c_size_t
GlobalSize.argtypes = [ctypes.wintypes.HGLOBAL]

EmptyClipboard = windll.user32.EmptyClipboard
EmptyClipboard.restype = BOOL
EmptyClipboard.argtypes = None

SetClipboardData = windll.user32.SetClipboardData
SetClipboardData.restype = HANDLE
SetClipboardData.argtypes = [UINT, HANDLE]

CloseClipboard = windll.user32.CloseClipboard
CloseClipboard.restype = BOOL
CloseClipboard.argtypes = None

GlobalFree = windll.kernel32.GlobalFree
GlobalFree.argtypes = [HANDLE]

GetClipboardData = windll.user32.GetClipboardData
GetClipboardData.argtypes = [ctypes.wintypes.UINT]
GetClipboardData.restype = ctypes.wintypes.HANDLE

EnumClipboardFormats = windll.user32.EnumClipboardFormats
EnumClipboardFormats.argtypes = [ctypes.wintypes.UINT]
EnumClipboardFormats.restype = ctypes.wintypes.UINT

BlockInput = windll.user32.BlockInput
BlockInput.argtypes = [BOOL]
BlockInput.restype = BOOL


class PyClipboardPlus:
    def __init__(self, maxlen: int | None = None):
        """
        Initializes the PyClipboardPlus object.

        Args:
            maxlen (int | None): Maximum length of the clipboard history. If None, there is no limit. Defaults to None.
        """
        self.maxlen = maxlen
        self.list_oem_texts = deque([], self.maxlen)
        self.list_unicode_texts = deque([], self.maxlen)
        self.list_cf_texts = deque([], self.maxlen)
        self.list_images = deque([], self.maxlen)
        self.list_files = deque([], self.maxlen)
        self.list_raw_all = deque([], self.maxlen)
        self.list_raw_known = deque([], self.maxlen)
        self.list_oem_texts_thread = None
        self.list_unicode_texts_thread = None
        self.list_cf_texts_thread = None
        self.list_images_thread = None
        self.list_files_thread = None
        self.list_raw_all_thread = None
        self.list_raw_known_thread = None

        self.list_oem_texts_thread_active = False
        self.list_unicode_texts_thread_active = False
        self.list_cf_texts_thread_active = False
        self.list_images_thread_active = False
        self.list_files_thread_active = False
        self.list_raw_all_thread_active = False
        self.list_raw_known_thread_active = False

    @staticmethod
    def get_handles_of_all_procs(print_results: bool = True) -> defaultdict:
        """
        Retrieves the handles of all processes.

        Args:
            print_results (bool): Whether to print the results or not. Defaults to True.

        Returns:
            defaultdict: A dictionary with the process IDs as keys and the corresponding window information as values.
        """
        r = groupBy(
            key=lambda i: i.pid,
            seq=get_window_infos(),
            continue_on_exceptions=True,
            withindex=False,
            withvalue=True,
        )
        if print_results:
            pprint(r)
        return r

    @staticmethod
    def show_all_ms_datatypes() -> dict:
        """
        Displays all available Microsoft data types.

        Returns:
            dict: A dictionary mapping data type names to their corresponding values.
        """
        pprint(clipboarddict)
        return clipboarddict

    @staticmethod
    def write_binary_data_to_clipboard(data: bytes, datatype: int) -> None:
        """
        Writes binary data to the clipboard.

        Args:
            data (bytes): The binary data to be written.
            datatype (int): The data type identifier.

        Returns:
            None
        """
        _write_data(data, datatype)

    @staticmethod
    def write_image_to_clipboard(image: Image.Image | np.ndarray | str) -> None:
        """
        Writes an image to the clipboard.

        Args:
            image (Image.Image | np.ndarray | str): The image to be written.

        Returns:
            None
        """
        image_to_clipboard(image)

    @staticmethod
    def write_oem_text_to_clipboard(text: bytes | str) -> None:
        """
        Writes OEM text to the clipboard.

        Args:
            text (bytes | str): The OEM text to be written.

        Returns:
            None
        """
        oem_text_to_clipboard(text)

    @staticmethod
    def write_unicode_text_to_clipboard(text: bytes | str) -> None:
        """
        Writes Unicode text to the clipboard.

        Args:
            text (bytes | str): The Unicode text to be written.

        Returns:
            None
        """
        unicode_text_to_clipboard(text)

    @staticmethod
    def write_cf_text_to_clipboard(text: bytes | str) -> None:
        """
        Writes CF text to the clipboard.

        Args:
            text (bytes | str): The CF text to be written.

        Returns:
            None
        """
        cf_text_to_clipboard(text)

    @staticmethod
    def write_file_list_to_clipboard(files: str | list | tuple) -> None:
        """
        Writes a file list to the clipboard.

        Args:
            files (str | list | tuple): The file list/single file to be written.

        Returns:
            None
        """
        if isinstance(files, str):
            files = [files]
        files = [os.path.normpath(x) for x in files]
        filelist_to_clipboard(files)

    @staticmethod
    def read_multiple_formats_from_clipboard_all(
        avoid_duplicates: bool = False,
    ) -> dict:
        """

        Reads multiple formats of data from the clipboard.

        Args:
            avoid_duplicates (bool): Whether to avoid duplicates in the results or not. Defaults to False.

        Returns:
            dict: A dictionary containing the data from the clipboard in different formats.
        """
        return deepcopy(
            read_all_raw_data_from_clipboard(
                only_standard_formats=False,
                forbidden_formats=(),
                allowed_formats=(),
                avoid_duplicates=avoid_duplicates,
            )
        )

    @staticmethod
    def read_multiple_formats_from_clipboard_known(
        avoid_duplicates: bool = False,
    ) -> dict:
        """
        Reads multiple known formats of data from the clipboard.

        Args:
            avoid_duplicates (bool): Whether to avoid duplicates in the results or not. Defaults to False.

        Returns:
            dict: A dictionary containing the data from the clipboard in different known formats.
        """
        return deepcopy(
            read_all_raw_data_from_clipboard(
                only_standard_formats=True,
                forbidden_formats=(),
                allowed_formats=(),
                avoid_duplicates=avoid_duplicates,
            )
        )

    @staticmethod
    def _get_forbidden(allowed_formats):
        return [x for x in allformatslist if x not in allowed_formats]

    @staticmethod
    def read_images_from_clipboard(
        nparray: bool = False, avoid_duplicates: bool = False
    ) -> Image.Image | np.ndarray | type(None):
        """
        Reads image data from the clipboard.

        Args:
            nparray (bool): If True, returns the image data as a NumPy array.
                            If False, returns the image data as a PIL Image object. (default: False)
            avoid_duplicates (bool): If True, avoids returning duplicate images. (default: False)

        Returns:
            Union[Image.Image, np.ndarray, None]: The image data from the clipboard.
                                                  If no image data is available or an error occurs, returns None.
                                                  If `nparray` is True, returns a NumPy array.
                                                  If `nparray` is False, returns a PIL Image object.
        """
        allowed_formats = ("CF_DIB",)
        forbidden_formats = PyClipboardPlus._get_forbidden(allowed_formats)
        try:
            bi = deepcopy(
                read_all_raw_data_from_clipboard(
                    only_standard_formats=True,
                    allowed_formats=allowed_formats,
                    forbidden_formats=forbidden_formats,
                    avoid_duplicates=avoid_duplicates,
                )[0]["data"]
            )
            if nparray:
                return np.array(deepcopy(bi))
            else:
                return deepcopy(bi)
        except Exception as fe:
            return None

    @staticmethod
    def read_file_list_from_clipboard(avoid_duplicates: bool = False) -> list:
        """
        Reads a list of file paths from the clipboard.

        Returns:
            List[str]: A list of file paths from the clipboard.
                       If no file paths are available or an error occurs, returns an empty list.
        """
        allowed_formats = ("CF_HDROP",)
        forbidden_formats = PyClipboardPlus._get_forbidden(allowed_formats)
        try:
            return deepcopy(
                read_all_raw_data_from_clipboard(
                    only_standard_formats=True,
                    allowed_formats=allowed_formats,
                    forbidden_formats=forbidden_formats,
                    avoid_duplicates=avoid_duplicates,
                )[0]["data"]
            )

        except Exception as fe:
            return []

    @staticmethod
    def _read_txt_formats(allowed_formats, avoid_duplicates=False) -> dict | None:
        forbidden_formats = PyClipboardPlus._get_forbidden(allowed_formats)
        try:
            return deepcopy(
                read_all_raw_data_from_clipboard(
                    only_standard_formats=True,
                    allowed_formats=allowed_formats,
                    forbidden_formats=forbidden_formats,
                    avoid_duplicates=avoid_duplicates,
                )[0]["data"]
            )

        except Exception as fe:
            return None

    @staticmethod
    def read_cf_text_from_clipboard(avoid_duplicates: bool = False) -> bytes:
        """
        Reads text from the clipboard

        Returns:
            Optional[bytes]: bytes
        """
        allowed_formats = ("CF_TEXT",)
        return deepcopy(PyClipboardPlus._read_txt_formats(
            allowed_formats, avoid_duplicates=avoid_duplicates
        ))

    @staticmethod
    def read_unicode_text_from_clipboard(avoid_duplicates: bool = False) -> str:
        """
        Reads Unicode text from the clipboard.

        Returns:
            Optional[str]: The Unicode text from the clipboard.
                           If no Unicode text is available or an error occurs, returns None.
        """
        allowed_formats = ("CF_UNICODETEXT",)
        return deepcopy(PyClipboardPlus._read_txt_formats(
            allowed_formats, avoid_duplicates=avoid_duplicates
        ))

    @staticmethod
    def read_oem_text_from_clipboard(avoid_duplicates: bool = False) -> str:
        """
        Reads OEM text from the clipboard.

        Args:
            avoid_duplicates (bool, optional): Flag to avoid reading duplicated text from the clipboard.
                                               Defaults to False.

        Returns:
            Optional[str]: The OEM text from the clipboard.
                           If no OEM text is available or an error occurs, returns None.
        """
        allowed_formats = ("CF_OEMTEXT",)
        return deepcopy(PyClipboardPlus._read_txt_formats(
            allowed_formats, avoid_duplicates=avoid_duplicates
        ))

    @staticmethod
    def paste_to_hwnd(
        hwnd: int, blockinput: bool = True, activate: bool = True
    ) -> None:
        """
        Pastes the contents of the clipboard to the specified window handle (hwnd).

        Args:
            hwnd (int): The window handle (hwnd) of the target window.
            blockinput (bool, optional): Whether to block user input during the paste operation. Defaults to True.
            activate (bool, optional): Whether to activate the target window before pasting. Defaults to True.

        Returns:
            None
        """
        if blockinput:
            if block_user_input():
                try:
                    mkey.send_keystrokes_to_hwnd(
                        handle=hwnd,
                        keystrokes="^V",
                        with_spaces=False,
                        with_tabs=False,
                        with_newlines=False,
                        activate_window_before=activate,
                    )
                finally:
                    unblock_user_input()
        else:
            mkey.send_keystrokes_to_hwnd(
                handle=hwnd,
                keystrokes="^V",
                with_spaces=False,
                with_tabs=False,
                with_newlines=False,
                activate_window_before=activate,
            )

    def _interval_check_images(self, interval, nparray, avoid_duplicates):
        while self.list_images_thread_active:
            rea = deepcopy(
                PyClipboardPlus.read_images_from_clipboard(
                    nparray=nparray, avoid_duplicates=avoid_duplicates
                )
            )
            if isinstance_tolerant(rea, None):
                pass
            else:
                self.list_images.append(rea)
            sleep(interval)

    def check_images_interval(
        self, interval: int = 1, nparray: bool = False, avoid_duplicates: bool = True
    ) -> None:
        """
        Starts a background thread that checks the clipboard for images at a specified interval.

        Args:
            interval (int, optional): The interval in seconds at which to check the clipboard for images. Defaults to 1.
            nparray (bool, optional): Specifies whether to return the images as NumPy arrays. Defaults to False.
            avoid_duplicates (bool, optional): Specifies whether to avoid duplicate images. Defaults to True.

        Returns:
            None (results are appended to self.list_unicode_texts)
        """

        self.list_images_thread_active = True
        self.list_images_thread = kthread.KThread(
            target=self._interval_check_images,
            name="_interval_check_images_function",
            args=(interval, nparray, avoid_duplicates),
        )
        self.list_images_thread.start()

    def stop_check_images_interval(self):
        self.list_images_thread_active = False
        sleep(1)
        while self.list_images_thread.is_alive():
            try:
                self.list_images_thread.kill()
            except Exception as fe:
                continue

    def _interval_check_unicode(self, interval, avoid_duplicates=False):
        while self.list_unicode_texts_thread_active:
            rea = deepcopy(
                PyClipboardPlus.read_unicode_text_from_clipboard(
                    avoid_duplicates=avoid_duplicates
                )
            )
            if isinstance_tolerant(rea, None):
                pass
            else:
                self.list_unicode_texts.append(rea)
            sleep(interval)

    def check_unicode_interval(
        self, interval: int = 1, avoid_duplicates: bool = True
    ) -> None:
        """
        Starts a background thread that checks the clipboard for Unicode texts at a specified interval.

        Args:
            interval (int, optional): The interval in seconds at which to check the clipboard for Unicode texts. Defaults to 1.
            avoid_duplicates (bool, optional): Specifies whether to avoid duplicate Unicode texts. Defaults to True.

        Returns:
            None (results are appended to self.list_unicode_texts)
        """
        self.list_unicode_texts_thread_active = True
        self.list_unicode_texts_thread = kthread.KThread(
            target=self._interval_check_unicode,
            name="_interval_check_unicode_function",
            args=(interval, avoid_duplicates),
        )
        self.list_unicode_texts_thread.start()

    def stop_check_unicode_interval(self):
        """
        Stops the background thread that checks the clipboard for Unicode texts.

        Returns:
            None
        """
        self.list_unicode_texts_thread_active = False
        sleep(1)
        while self.list_unicode_texts_thread.is_alive():
            try:
                self.list_unicode_texts_thread.kill()
            except Exception as fe:
                continue

    def _interval_check_oem(self, interval, avoid_duplicates=False):
        while self.list_oem_texts_thread_active:
            rea = deepcopy(
                PyClipboardPlus.read_oem_text_from_clipboard(
                    avoid_duplicates=avoid_duplicates
                )
            )
            if isinstance_tolerant(rea, None):
                pass
            else:
                self.list_oem_texts.append(rea)
            sleep(interval)

    def check_oem_interval(
        self, interval: int = 1, avoid_duplicates: bool = True
    ) -> None:
        """
        Starts a background thread that checks the clipboard for OEM texts at a specified interval.

        Args:
            interval (int, optional): The interval in seconds at which to check the clipboard for OEM texts. Defaults to 1.
            avoid_duplicates (bool, optional): Specifies whether to avoid duplicate OEM texts. Defaults to True.

        Returns:
            None  (results are appended to self.list_cf_texts)
        """
        self.list_oem_texts_thread_active = True
        self.list_oem_texts_thread = kthread.KThread(
            target=self._interval_check_oem,
            name="_interval_check_oem_function",
            args=(interval, avoid_duplicates),
        )
        self.list_oem_texts_thread.start()

    def stop_check_oem_interval(self):
        """
        Stops the background thread that checks the clipboard for OEM texts.

        Returns:
            None
        """
        self.list_oem_texts_thread_active = False
        sleep(1)
        while self.list_oem_texts_thread.is_alive():
            try:
                self.list_oem_texts_thread.kill()
            except Exception as fe:
                continue

    def _interval_check_cf(self, interval, avoid_duplicates=False):
        while self.list_cf_texts_thread_active:
            rea = deepcopy(
                PyClipboardPlus.read_cf_text_from_clipboard(
                    avoid_duplicates=avoid_duplicates
                )
            )
            if isinstance_tolerant(rea, None):
                pass
            else:
                self.list_cf_texts.append(rea)
            sleep(interval)

    def check_cf_interval(
        self, interval: int = 1, avoid_duplicates: bool = True
    ) -> None:
        """
        Starts a background thread that checks the clipboard for CF texts at a specified interval.

        Args:
            interval (int, optional): The interval in seconds at which to check the clipboard for CF texts. Defaults to 1.
            avoid_duplicates (bool, optional): Specifies whether to avoid duplicate CF texts. Defaults to True.

        Returns:
            None  (results are appended to self.list_cf_texts_thread)
        """
        self.list_cf_texts_thread_active = True
        self.list_cf_texts_thread = kthread.KThread(
            target=self._interval_check_cf,
            name="_interval_check_cf_function",
            args=(interval, avoid_duplicates),
        )
        self.list_cf_texts_thread.start()

    def stop_check_cf_interval(self):
        """
        Stops the background thread that checks the clipboard for CF texts.

        Returns:
            None
        """
        self.list_cf_texts_thread_active = False
        sleep(1)
        while self.list_cf_texts_thread.is_alive():
            try:
                self.list_cf_texts_thread.kill()
            except Exception as fe:
                continue

    def _interval_check_files(self, interval, avoid_duplicates=False):
        while self.list_files_thread_active:
            rea = deepcopy(
                PyClipboardPlus.read_file_list_from_clipboard(
                    avoid_duplicates=avoid_duplicates
                )
            )
            if isinstance_tolerant(rea, None):
                pass
            elif len(rea) == 0:
                pass
            else:
                self.list_files.append(rea)
            sleep(interval)

    def check_files_interval(
        self, interval: int = 1, avoid_duplicates: bool = True
    ) -> None:
        """
        Starts a background thread that checks the clipboard for file lists at a specified interval.

        Args:
            interval (int, optional): The interval in seconds at which to check the clipboard for file lists. Defaults to 1.
            avoid_duplicates (bool, optional): Specifies whether to avoid duplicate file lists. Defaults to True.

        Returns:
            None
        """
        self.list_files_thread_active = True
        self.list_files_thread = kthread.KThread(
            target=self._interval_check_files,
            name="_interval_check_filelist_function",
            args=(interval, avoid_duplicates),
        )
        self.list_files_thread.start()

    def stop_check_files_interval(self):
        """
        Stops the background thread that checks the clipboard for file lists.

        Returns:
            None
        """
        self.list_files_thread_active = False
        sleep(1)
        while self.list_files_thread.is_alive():
            try:
                self.list_files_thread.kill()
            except Exception as fe:
                continue


def block_user_input():
    return BlockInput(True)


def unblock_user_input():
    return BlockInput(False)


globvars = sys.modules[__name__]
globvars.duplicated_stuff = []


def _write_data(data, datatype):
    try:
        datasize = len(data)
        hData = GlobalAlloc(GHND | GMEM_SHARE, datasize)
        pData = GlobalLock(hData)
        ctypes.memmove(pData, data, datasize)
        GlobalUnlock(hData)
        OpenClipboard(None)
        EmptyClipboard()
        SetClipboardData(datatype, pData)
        CloseClipboard()
    except Exception as fe:
        try:
            CloseClipboard()
        except Exception as fa:
            pass


def image_to_clipboard(image):
    if "numpy" in str(type(image)):
        image = Image.fromarray(image.copy())
    elif isinstance(image, str):
        if os.path.exists(image):
            image = Image.open(image)

    output = BytesIO()
    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    _write_data(data, CF_DIB)


def oem_text_to_clipboard(text):
    if isinstance(text, str):
        text = text.encode(sys.getfilesystemencoding())
    text = text + b"\x00"
    _write_data(text, CF_OEMTEXT)


def unicode_text_to_clipboard(text):
    if isinstance(text, str):
        text = text.encode("utf-16-le")
    text = text + b"\x00"
    _write_data(text, CF_UNICODETEXT)


def cf_text_to_clipboard(text):
    if isinstance(text, str):
        text = text.encode()
    text = text + b"\x00"
    _write_data(text, CF_TEXT)


def filelist_to_clipboard(filelist):
    text = b"\x00\x00".join([x.encode("utf-16-le") for x in filelist])  #
    text = (
        b"\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
        + text.rstrip(b"\x00")
    )
    _write_data(text, CF_HDROP)


def read_all_raw_data_from_clipboard(
    only_standard_formats=True,
    forbidden_formats=(),
    allowed_formats=(),
    avoid_duplicates=False,
):
    indi = 0
    allelements = {}
    alfo = []
    OpenClipboard(None)
    formatx = EnumClipboardFormats(0)
    alfo.append(formatx)
    while formatx:
        try:
            formatx = EnumClipboardFormats(formatx)
            try:
                dty = clipboarddict.get(formatx)["constant"]
            except Exception:
                pass
            if only_standard_formats:
                if formatx not in clipboarddict:
                    continue
            if dty in forbidden_formats:
                continue
            if allowed_formats:
                if dty not in allowed_formats:
                    continue
            alfo.append(formatx)
        except Exception:
            continue
    CloseClipboard()

    for formatx in alfo:
        print(formatx)
        try:
            OpenClipboard(None)
            clipboard_data = GetClipboardData(formatx)
            CloseClipboard()

            if clipboard_data:
                buffer_size = GlobalSize(clipboard_data)
                buffer = ctypes.create_string_buffer(buffer_size)

                ctypes.memmove(buffer, clipboard_data, buffer_size)
                bytes_data = deepcopy(buffer.raw)
                if avoid_duplicates:
                    hash_ = hash(bytes_data)
                    if hash_ in globvars.duplicated_stuff:
                        continue
                    else:
                        globvars.duplicated_stuff.append(hash_)
                dty = formatx

                try:
                    dty = clipboarddict.get(formatx)["constant"]
                except Exception:
                    pass
                addtodict = True
                if only_standard_formats:
                    if formatx not in clipboarddict:
                        addtodict = False
                if dty in forbidden_formats:
                    addtodict = False
                if allowed_formats:
                    if dty not in allowed_formats:
                        addtodict = False
                if addtodict:
                    allelements[indi] = {
                        "data": deepcopy(np.array(deepcopy(bytes_data))),
                        "datatype": dty,
                    }
                    if dty == "CF_DIB":
                        data = io.BytesIO(allelements[indi]["data"].tobytes())
                        allelements[indi]["data"] = BmpImagePlugin.DibImageFile(data)

                    elif dty == "CF_HDROP":
                        allelements[indi]["data"] = [
                            x
                            for x in allelements[indi]["data"]
                            .tobytes()
                            .decode("utf-16-le")
                            .split("\x00")
                            if os.path.exists(x)
                        ]
                    elif dty == "CF_OEMTEXT":
                        allelements[indi]["data"] = (
                            allelements[indi]["data"]
                            .tobytes()
                            .decode(sys.getfilesystemencoding()).rstrip('\x00')
                        )
                    elif dty == "CF_UNICODETEXT":
                        print(allelements[indi]["data"])
                        try:
                            allelements[indi]["data"] = (
                                (allelements[indi]["data"].tobytes() + b'\x00').decode("utf-16-le").rstrip('\x00')
                            )
                        except Exception:
                            allelements[indi]["data"] = (
                                allelements[indi]["data"].tobytes().decode("utf-16-le").rstrip('\x00')
                            )
                    elif dty == "CF_TEXT":
                        allelements[indi]["data"] = allelements[indi]["data"].tobytes()
                    indi += 1
        except Exception as fe:
            indi += 1
            continue
    return allelements


def convert_to_normal_dict(di):
    if isinstance_tolerant(di, defaultdict):
        di = {k: convert_to_normal_dict(v) for k, v in di.items()}
    return di


def groupBy(key, seq, continue_on_exceptions=True, withindex=True, withvalue=True):
    indexcounter = -1

    def execute_f(k, v):
        nonlocal indexcounter
        indexcounter += 1
        try:
            return k(v)
        except Exception as fa:
            if continue_on_exceptions:
                return "EXCEPTION: " + str(fa)
            else:
                raise fa

    # based on https://stackoverflow.com/a/60282640/15096247
    if withvalue:
        return convert_to_normal_dict(
            reduce(
                lambda grp, val: grp[execute_f(key, val)].append(
                    val if not withindex else (indexcounter, val)
                )
                or grp,
                seq,
                defaultdict(list),
            )
        )
    return convert_to_normal_dict(
        reduce(
            lambda grp, val: grp[execute_f(key, val)].append(indexcounter) or grp,
            seq,
            defaultdict(list),
        )
    )


clipboarddict = {
    2: {"constant": "CF_BITMAP", "description": "A handle to a bitmap (HBITMAP)."},
    8: {
        "constant": "CF_DIB",
        "description": "A memory object containing a BITMAPINFO structure followed by the bitmap bits.",
    },
    17: {
        "constant": "CF_DIBV5",
        "description": "A memory object containing a BITMAPV5HEADER structure followed by the bitmap color space information and the bitmap bits.",
    },
    5: {"constant": "CF_DIF", "description": "Software Arts' Data Interchange Format."},
    130: {
        "constant": "CF_DSPBITMAP",
        "description": "Bitmap display format associated with a private format. The hMem parameter must be a handle to data that can be displayed in bitmap format in lieu of the privately formatted data.",
    },
    142: {
        "constant": "CF_DSPENHMETAFILE",
        "description": "Enhanced metafile display format associated with a private format. The hMem parameter must be a handle to data that can be displayed in enhanced metafile format in lieu of the privately formatted data.",
    },
    131: {
        "constant": "CF_DSPMETAFILEPICT",
        "description": "Metafile-picture display format associated with a private format. The hMem parameter must be a handle to data that can be displayed in metafile-picture format in lieu of the privately formatted data.",
    },
    129: {
        "constant": "CF_DSPTEXT",
        "description": "Text display format associated with a private format. The hMem parameter must be a handle to data that can be displayed in text format in lieu of the privately formatted data.",
    },
    14: {
        "constant": "CF_ENHMETAFILE",
        "description": "A handle to an enhanced metafile (HENHMETAFILE).",
    },
    768: {
        "constant": "CF_GDIOBJFIRST",
        "description": "Start of a range of integer values for application-defined GDI object clipboard formats. The end of the range is CF_GDIOBJLAST.  Handles associated with clipboard formats in this range are not automatically deleted using the GlobalFree function when the clipboard is emptied. Also, when using values in this range, the hMem parameter is not a handle to a GDI object, but is a handle allocated by the GlobalAlloc function with the GMEM_MOVEABLE flag.",
    },
    1023: {"constant": "CF_GDIOBJLAST", "description": "See CF_GDIOBJFIRST."},
    15: {
        "constant": "CF_HDROP",
        "description": "A handle to type HDROP that identifies a list of files. An application can retrieve information about the files by passing the handle to the DragQueryFile function.",
    },
    16: {
        "constant": "CF_LOCALE",
        "description": "The data is a handle (HGLOBAL) to the locale identifier (LCID) associated with text in the clipboard. When you close the clipboard, if it contains CF_TEXT data but no CF_LOCALE data, the system automatically sets the CF_LOCALE format to the current input language. You can use the CF_LOCALE format to associate a different locale with the clipboard text. An application that pastes text from the clipboard can retrieve this format to determine which character set was used to generate the text.  Note that the clipboard does not support plain text in multiple character sets. To achieve this, use a formatted text data type such as RTF instead.  The system uses the code page associated with CF_LOCALE to implicitly convert from CF_TEXT to CF_UNICODETEXT. Therefore, the correct code page table is used for the conversion.",
    },
    3: {
        "constant": "CF_METAFILEPICT",
        "description": "Handle to a metafile picture format as defined by the METAFILEPICT structure. When passing a CF_METAFILEPICT handle by means of DDE, the application responsible for deleting hMem should also free the metafile referred to by the CF_METAFILEPICT handle.",
    },
    7: {
        "constant": "CF_OEMTEXT",
        "description": "Text format containing characters in the OEM character set. Each line ends with a carriage return/linefeed (CR-LF) combination. A null character signals the end of the data.",
    },
    128: {
        "constant": "CF_OWNERDISPLAY",
        "description": "Owner-display format. The clipboard owner must display and update the clipboard viewer window, and receive the WM_ASKCBFORMATNAME, WM_HSCROLLCLIPBOARD, WM_PAINTCLIPBOARD, WM_SIZECLIPBOARD, and WM_VSCROLLCLIPBOARD messages. The hMem parameter must be NULL.",
    },
    9: {
        "constant": "CF_PALETTE",
        "description": "Handle to a color palette. Whenever an application places data in the clipboard that depends on or assumes a color palette, it should place the palette on the clipboard as well.  If the clipboard contains data in the CF_PALETTE (logical color palette) format, the application should use the SelectPalette and RealizePalette functions to realize (compare) any other data in the clipboard against that logical palette.  When displaying clipboard data, the clipboard always uses as its current palette any object on the clipboard that is in the CF_PALETTE format.",
    },
    10: {
        "constant": "CF_PENDATA",
        "description": "Data for the pen extensions to the Microsoft Windows for Pen Computing.",
    },
    512: {
        "constant": "CF_PRIVATEFIRST",
        "description": "Start of a range of integer values for private clipboard formats. The range ends with CF_PRIVATELAST. Handles associated with private clipboard formats are not freed automatically; the clipboard owner must free such handles, typically in response to the WM_DESTROYCLIPBOARD message.",
    },
    767: {"constant": "CF_PRIVATELAST", "description": "See CF_PRIVATEFIRST."},
    11: {
        "constant": "CF_RIFF",
        "description": "Represents audio data more complex than can be represented in a CF_WAVE standard wave format.",
    },
    4: {"constant": "CF_SYLK", "description": "Microsoft Symbolic Link (SYLK) format."},
    1: {
        "constant": "CF_TEXT",
        "description": "Text format. Each line ends with a carriage return/linefeed (CR-LF) combination. A null character signals the end of the data. Use this format for ANSI text.",
    },
    6: {"constant": "CF_TIFF", "description": "Tagged-image file format."},
    13: {
        "constant": "CF_UNICODETEXT",
        "description": "Unicode text format. Each line ends with a carriage return/linefeed (CR-LF) combination. A null character signals the end of the data.",
    },
    12: {
        "constant": "CF_WAVE",
        "description": "Represents audio data in one of the standard wave formats, such as 11 kHz or 22 kHz PCM.",
    },
}
allformatslist = [x[1]["constant"] for x in clipboarddict.items()]


