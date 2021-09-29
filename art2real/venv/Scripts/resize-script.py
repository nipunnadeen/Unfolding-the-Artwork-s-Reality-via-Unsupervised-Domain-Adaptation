#!C:\Users\Asus\Desktop\Art2Real\art2real\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'resize==0.1.0','console_scripts','resize'
__requires__ = 'resize==0.1.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('resize==0.1.0', 'console_scripts', 'resize')()
    )
