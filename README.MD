# Clipboard - Write/Read - supports Images, UTF-8, bytes, file lists, and more! (Windows only)

## pip install ctpaperclip

#### Tested against Windows 10 / Python 3.10 / Anaconda


This module is designed to cater to the needs of the people who frequently work with the clipboard and 
require the ability to manipulate and transfer data in multiple formats. While other clipboard libraries 
like Pyperclip focus primarily on text-based operations, this module offers a broader range of capabilities.

## Here are some specific advantages of using this module:

Support for Multiple Formats: 

Unlike many clipboard libraries that are limited to handling text, this module supports various data formats. 
It allows you to read and write not only text but also images and file lists. This versatility 
enables seamless interaction with different types of data on the clipboard.

### Automatic Format Recognition: 

One of the notable features of this module is its ability to automatically recognize the format of data 
present on the clipboard. This eliminates the need for manual format detection and simplifies the development 
process. The module intelligently detects whether the clipboard contains text, images, or file lists, 
allowing you to focus on utilizing the data rather than dealing with format identification.

### Integration with NumPy and PIL: 

This module provides seamless integration with popular libraries such as NumPy and PIL (Python Imaging Library). 
It allows you to read images from the clipboard as NumPy arrays or PIL images, providing a convenient interface 
for image processing and manipulation. This integration enhances the capabilities of the module and enables 
advanced image-related operations.

### Writing Data in Multiple Formats: 

In addition to reading data from the clipboard, this module also allows you to write data back to the clipboard 
in various formats. Whether you need to copy modified text, processed images, or a list of files, the module 
supports writing data in the appropriate format. This flexibility empowers you to manipulate and transfer data 
in the desired format without any constraints.



```python

from time import sleep

from PIL import Image

from ctpaperclip import PyClipboardPlus

if __name__ == "__main__":
    # Create an instance
    pycl = PyClipboardPlus()

    allp = pycl.get_handles_of_all_procs()
    # 11656: [ WindowInfo(pid=11656, title='MSCTFIME UI', windowtext='MSCTFIME UI', hwnd=68492, length=12, tid=23292, status='invisible', coords_client=(0, 0, 0, 0), dim_client=(0, 0), coords_win=(0, 0, 0, 0), dim_win=(0, 0), class_name='MSCTFIME UI', path='C:\\Windows\\System32\\notepad.exe'),
    #          WindowInfo(pid=11656, title='Notepad', windowtext='*sdfsdfsdfsfdsfsdfsfsdf.txt - Notepad', hwnd=5507962, length=38, tid=23292, status='visible', coords_client=(0, 895, 0, 678), dim_client=(895, 678), coords_win=(577, 1488, 288, 1025), dim_win=(911, 737), class_name='Notepad', path='C:\\Windows\\System32\\notepad.exe')],
    #          ....]

    cldi = pycl.show_all_ms_datatypes()
    # {2: {'constant': 'CF_BITMAP',
    #   'description': 'A handle to a bitmap (HBITMAP).'},
    #  8: {'constant': 'CF_DIB',
    #   'description': 'A memory object containing a BITMAPINFO structure followed by the bitmap bits.'},
    #  17: {'constant': 'CF_DIBV5',
    # ...

    image = Image.new("RGB", (100, 200), (255, 100, 0))
    pycl.write_image_to_clipboard(image)
    sleep(1)
    imagefromcli = pycl.read_images_from_clipboard(nparray=False)
    # imagefromcli
    # Out[13]: <PIL.BmpImagePlugin.DibImageFile image mode=RGB size=100x200>

    image = Image.new("RGB", (100, 200), (255, 100, 0))
    pycl.write_image_to_clipboard(image)
    imagefromcli = pycl.read_images_from_clipboard(nparray=True)
    print(imagefromcli)
    # [[[255 100   0]
    #   [255 100   0]
    #   [255 100   0]
    #   ...
    #   [255 100   0]
    #   [255 100   0]
    #   [255 100   0]]
    # ...

    pycl.write_oem_text_to_clipboard("Hamnösx")
    pycl.read_oem_text_from_clipboard()
    # Out[7]: 'Hamnösx'

    pycl.write_cf_text_to_clipboard("Hamnösx")
    pycl.read_cf_text_from_clipboard()

    # Out[4]: b'Hamn\xc3\xb6sx\x00'

    pycl.write_unicode_text_to_clipboard("Hamnösx")
    pycl.read_unicode_text_from_clipboard()
    # Out[4]: 'Hamnösx'

    sleep(1)
    pycl.paste_to_hwnd(5507962, blockinput=True, activate=True)  # notepad hwnd 5507962
    sleep(1)
    pycl.check_files_interval()
    pycl.list_files
    # Out[3]:
    # deque([['C:\\ProgramData\\anaconda3\\envs\\dfdir\\imageclip.py',
    #         'C:\\ProgramData\\anaconda3\\envs\\dfdir\\ctypesstuff.py',
    #         'C:\\ProgramData\\anaconda3\\envs\\dfdir\\g2.py',
    #         'C:\\ProgramData\\anaconda3\\envs\\dfdir\\g.py',
    #         'C:\\ProgramData\\anaconda3\\envs\\dfdir\\pypigitupload.py',
    #         'C:\\ProgramData\\anaconda3\\envs\\dfdir\\suicideprocxxxxxxxxxxxxxxxx.py']])
    sleep(1)

    pycl.stop_check_files_interval()
    sleep(1)

    pycl.write_file_list_to_clipboard(
        [
            "C:\\ProgramData\\anaconda3\\envs\\dfdir\\imageclip.py",
            "C:\\ProgramData\\anaconda3\\envs\\dfdir\\ctypesstuff.py",
            "C:\\ProgramData\\anaconda3\\envs\\dfdir\\g2.py",
            "C:\\ProgramData\\anaconda3\\envs\\dfdir\\g.py",
            "C:\\ProgramData\\anaconda3\\envs\\dfdir\\pypigitupload.py",
            "C:\\ProgramData\\anaconda3\\envs\\dfdir\\suicideprocxxxxxxxxxxxxxxxx.py",
        ]
    )
    # Create a new folder with the explorer, and press ctrl+v,
    # The files will be copied to that folder.
    pycl.check_cf_interval()
    sleep(1)

    pycl.stop_check_cf_interval()
    pycl.list_cf_texts
    # Out[14]: deque([b"ist_raw_all_thread': None,\r\n 'list_raw_known_thread': None,\r\n 'list_oem_texts_thread_active': False,\r\n 'list_unicode_texts_thread_active': False,\r\n 'list_cf_texts_thread_active': False,\r\n 'list_images_thread_active': False,\r\n 'list_files_thread_active': False,\r\n 'list_raw_all_thread_active': False,\r\n 'list_raw_known_thread_active': False\x00"])

    pycl.check_oem_interval()
    pycl.stop_check_oem_interval()
    pycl.list_oem_texts
    # Out[18]:
    # deque(['check_oem_interval\x00',
    #        '\r\n        Args:\r\n            interval (int, optional): The interval in seconds at which to check the clipboard for OEM texts. Defaults to 1.\r\n            avoid_duplicates (bool, optional): Specifies whether to avoid duplicate OEM texts. Defaults to True.\r\n\r\n        Returns:\r\n            None  (results are appended to self.list_cf_texts)\x00',
    #        'stop_check_oem_interval\x00',
    #        'list_cf_texts\x00',
    #        " 'list_cf_texts_thread': None,\r\n 'list_images_thread': None,\r\n 'list_files_thread': <KThread(_interval_check_filelist_function, stopped 26836)>,\r\n 'list_raw_all_thread': None,\r\n 'list_raw_known_thread': None,\r\n 'list_oem_texts_thread_active': False,\r\n 'list_unicode_texts_thread_active': False,\r\n 'list_cf_texts_thread_active': False,\r\n 'list_images_thread_active': False,\r\n 'list_files_thread_active': False,\r\n 'list_raw_all_thread_active': Fal\x00",
    #        'Out[14]: deque([b"ist_raw_all_thread\': None,\\r\\n \'list_raw_known_thread\': None,\\r\\n \'list_oem_texts_thread_active\': False,\\r\\n \'list_unicode_texts_thread_active\': False,\\r\\n \'list_cf_texts_thread_active\': False,\\r\\n \'list_images_thread_active\': False,\\r\\n \'list_files_thread_active\': False,\\r\\n \'list_raw_all_thread_active\': False,\\r\\n \'list_raw_known_thread_active\': False\\x00"])\r\npycl.stop_check_cf_interval()\r\npycl.check_oem_interval()\x00'])

    pycl.check_images_interval(nparray=False)
    pycl.stop_check_images_interval()
    pycl.list_images
    # Out[21]: deque([<PIL.BmpImagePlugin.DibImageFile image mode=RGB size=519x243>])

    pycl.list_images.clear()
    pycl.check_images_interval(nparray=True)
    pycl.stop_check_images_interval()
    pycl.list_images
    # Out[26]:
    # deque([array([[[43, 43, 43],
    #                [43, 43, 43],
    #                [43, 43, 43],
    #                ...,
    #                [43, 43, 43],
    #                [43, 43, 43],
    #                [43, 43, 43]],

    pycl.check_unicode_interval()
    pycl.stop_check_unicode_interval()
    pycl.list_unicode_texts
    # Out[29]:
    # deque(['pycl.check_unicode_interval()\x00',
    #        '     self.list_unicode_texts_thread_active = True\r\n        self.list_unicode_texts_thread = kthread.KThread(\r\n            target=self._interval_check_unicode,\r\n            name="_interval_check_unicode_function",\r\n            args=(interval, avoid_du\x00',
    #        '\r\n            interval (int, optional): The interval in seconds at which to check the clipboard for Unicode texts. Defaults to 1.\r\n            avoid_duplicates (bool, optional): Specifies whether to avoid duplicate Unicode texts. Defaults to True.\r\n\x00',
    #        'stop_check_unicode_interval\x00'])



```



```python

class PyClipboardPlus(builtins.object)
 |  PyClipboardPlus(maxlen: int | None = None)
 |  
 |  
 |  __init__(self, maxlen: int | None = None)
 |      Initializes the PyClipboardPlus object.
 |      
 |      Args:
 |          maxlen (int | None): Maximum length of the clipboard history. If None, there is no limit. Defaults to None.
 |  
 |  check_cf_interval(self, interval: int = 1, avoid_duplicates: bool = True) -> None
 |      Starts a background thread that checks the clipboard for CF texts at a specified interval.
 |      
 |      Args:
 |          interval (int, optional): The interval in seconds at which to check the clipboard for CF texts. Defaults to 1.
 |          avoid_duplicates (bool, optional): Specifies whether to avoid duplicate CF texts. Defaults to True.
 |      
 |      Returns:
 |          None  (results are appended to self.list_cf_texts_thread)
 |  
 |  check_files_interval(self, interval: int = 1, avoid_duplicates: bool = True) -> None
 |      Starts a background thread that checks the clipboard for file lists at a specified interval.
 |      
 |      Args:
 |          interval (int, optional): The interval in seconds at which to check the clipboard for file lists. Defaults to 1.
 |          avoid_duplicates (bool, optional): Specifies whether to avoid duplicate file lists. Defaults to True.
 |      
 |      Returns:
 |          None
 |  
 |  check_images_interval(self, interval: int = 1, nparray: bool = False, avoid_duplicates: bool = True) -> None
 |      Starts a background thread that checks the clipboard for images at a specified interval.
 |      
 |      Args:
 |          interval (int, optional): The interval in seconds at which to check the clipboard for images. Defaults to 1.
 |          nparray (bool, optional): Specifies whether to return the images as NumPy arrays. Defaults to False.
 |          avoid_duplicates (bool, optional): Specifies whether to avoid duplicate images. Defaults to True.
 |      
 |      Returns:
 |          None (results are appended to self.list_unicode_texts)
 |  
 |  check_oem_interval(self, interval: int = 1, avoid_duplicates: bool = True) -> None
 |      Starts a background thread that checks the clipboard for OEM texts at a specified interval.
 |      
 |      Args:
 |          interval (int, optional): The interval in seconds at which to check the clipboard for OEM texts. Defaults to 1.
 |          avoid_duplicates (bool, optional): Specifies whether to avoid duplicate OEM texts. Defaults to True.
 |      
 |      Returns:
 |          None  (results are appended to self.list_cf_texts)
 |  
 |  check_unicode_interval(self, interval: int = 1, avoid_duplicates: bool = True) -> None
 |      Starts a background thread that checks the clipboard for Unicode texts at a specified interval.
 |      
 |      Args:
 |          interval (int, optional): The interval in seconds at which to check the clipboard for Unicode texts. Defaults to 1.
 |          avoid_duplicates (bool, optional): Specifies whether to avoid duplicate Unicode texts. Defaults to True.
 |      
 |      Returns:
 |          None (results are appended to self.list_unicode_texts)
 |  
 |  stop_check_cf_interval(self)
 |      Stops the background thread that checks the clipboard for CF texts.
 |      
 |      Returns:
 |          None
 |  
 |  stop_check_files_interval(self)
 |      Stops the background thread that checks the clipboard for file lists.
 |      
 |      Returns:
 |          None
 |  
 |  stop_check_images_interval(self)
 |  
 |  stop_check_oem_interval(self)
 |      Stops the background thread that checks the clipboard for OEM texts.
 |      
 |      Returns:
 |          None
 |  
 |  stop_check_unicode_interval(self)
 |      Stops the background thread that checks the clipboard for Unicode texts.
 |      
 |      Returns:
 |          None
 |  
 |  ----------------------------------------------------------------------
 |  Static methods:
 |  
 |  get_handles_of_all_procs(print_results: bool = True) -> collections.defaultdict
 |      Retrieves the handles of all processes.
 |      
 |      Args:
 |          print_results (bool): Whether to print the results or not. Defaults to True.
 |      
 |      Returns:
 |          defaultdict: A dictionary with the process IDs as keys and the corresponding window information as values.
 |  
 |  paste_to_hwnd(hwnd: int, blockinput: bool = True, activate: bool = True) -> None
 |      Pastes the contents of the clipboard to the specified window handle (hwnd).
 |      
 |      Args:
 |          hwnd (int): The window handle (hwnd) of the target window.
 |          blockinput (bool, optional): Whether to block user input during the paste operation. Defaults to True.
 |          activate (bool, optional): Whether to activate the target window before pasting. Defaults to True.
 |      
 |      Returns:
 |          None
 |  
 |  read_cf_text_from_clipboard(avoid_duplicates: bool = False) -> bytes
 |      Reads text from the clipboard
 |      
 |      Returns:
 |          Optional[bytes]: bytes
 |  
 |  read_file_list_from_clipboard(avoid_duplicates: bool = False) -> list
 |      Reads a list of file paths from the clipboard.
 |      
 |      Returns:
 |          List[str]: A list of file paths from the clipboard.
 |                     If no file paths are available or an error occurs, returns an empty list.
 |  
 |  read_images_from_clipboard(nparray: bool = False, avoid_duplicates: bool = False) -> PIL.Image.Image | numpy.ndarray | None
 |      Reads image data from the clipboard.
 |      
 |      Args:
 |          nparray (bool): If True, returns the image data as a NumPy array.
 |                          If False, returns the image data as a PIL Image object. (default: False)
 |          avoid_duplicates (bool): If True, avoids returning duplicate images. (default: False)
 |      
 |      Returns:
 |          Union[Image.Image, np.ndarray, None]: The image data from the clipboard.
 |                                                If no image data is available or an error occurs, returns None.
 |                                                If `nparray` is True, returns a NumPy array.
 |                                                If `nparray` is False, returns a PIL Image object.
 |  
 |  read_multiple_formats_from_clipboard_all(avoid_duplicates: bool = False) -> dict
 |      Reads multiple formats of data from the clipboard.
 |      
 |      Args:
 |          avoid_duplicates (bool): Whether to avoid duplicates in the results or not. Defaults to False.
 |      
 |      Returns:
 |          dict: A dictionary containing the data from the clipboard in different formats.
 |  
 |  read_multiple_formats_from_clipboard_known(avoid_duplicates: bool = False) -> dict
 |      Reads multiple known formats of data from the clipboard.
 |      
 |      Args:
 |          avoid_duplicates (bool): Whether to avoid duplicates in the results or not. Defaults to False.
 |      
 |      Returns:
 |          dict: A dictionary containing the data from the clipboard in different known formats.
 |  
 |  read_oem_text_from_clipboard(avoid_duplicates: bool = False) -> str
 |      Reads OEM text from the clipboard.
 |      
 |      Args:
 |          avoid_duplicates (bool, optional): Flag to avoid reading duplicated text from the clipboard.
 |                                             Defaults to False.
 |      
 |      Returns:
 |          Optional[str]: The OEM text from the clipboard.
 |                         If no OEM text is available or an error occurs, returns None.
 |  
 |  read_unicode_text_from_clipboard(avoid_duplicates: bool = False) -> str
 |      Reads Unicode text from the clipboard.
 |      
 |      Returns:
 |          Optional[str]: The Unicode text from the clipboard.
 |                         If no Unicode text is available or an error occurs, returns None.
 |  
 |  show_all_ms_datatypes() -> dict
 |      Displays all available Microsoft data types.
 |      
 |      Returns:
 |          dict: A dictionary mapping data type names to their corresponding values.
 |  
 |  write_binary_data_to_clipboard(data: bytes, datatype: int) -> None
 |      Writes binary data to the clipboard.
 |      
 |      Args:
 |          data (bytes): The binary data to be written.
 |          datatype (int): The data type identifier.
 |      
 |      Returns:
 |          None
 |  
 |  write_cf_text_to_clipboard(text: bytes | str) -> None
 |      Writes CF text to the clipboard.
 |      
 |      Args:
 |          text (bytes | str): The CF text to be written.
 |      
 |      Returns:
 |          None
 |  
 |  write_file_list_to_clipboard(files: str | list | tuple) -> None
 |      Writes a file list to the clipboard.
 |      
 |      Args:
 |          files (str | list | tuple): The file list/single file to be written.
 |      
 |      Returns:
 |          None
 |  
 |  write_image_to_clipboard(image: PIL.Image.Image | numpy.ndarray | str) -> None
 |      Writes an image to the clipboard.
 |      
 |      Args:
 |          image (Image.Image | np.ndarray | str): The image to be written.
 |      
 |      Returns:
 |          None
 |  
 |  write_oem_text_to_clipboard(text: bytes | str) -> None
 |      Writes OEM text to the clipboard.
 |      
 |      Args:
 |          text (bytes | str): The OEM text to be written.
 |      
 |      Returns:
 |          None
 |  
 |  write_unicode_text_to_clipboard(text: bytes | str) -> None
 |      Writes Unicode text to the clipboard.
 |      
 |      Args:
 |          text (bytes | str): The Unicode text to be written.
 |      
 |      Returns:
 |          None
 |  
 ```