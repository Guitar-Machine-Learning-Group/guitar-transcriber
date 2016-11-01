# guitar-transcriber

Guitar transcription using ML

## Dependencies

* NumPy
* SciPy
* Librosa
* pretty_midi

```
sudo pip install -r requirements.txt
```

If Python3

Might need to change Line 84 in "/usr/local/lib/python3.5/dist-packages/pretty_midi/utilities.py"
```
    ur'^(?P<key>[ABCDEFGabcdefg])'
```
to
```
    '^(?P<key>[ABCDEFGabcdefg])'
```

And add following code in the beginning of "/usr/local/lib/python3.5/dist-packages/pretty_midi/pretty_midi.py"
```
try:
    unicode = unicode
except NameError:
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring
```
