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
Might need to change Line 84 in "/usr/local/lib/python3.5/dist-packages/pretty_midi/utilities.py"
```
    ur'^(?P<key>[ABCDEFGabcdefg])'
```
to
```
    '^(?P<key>[ABCDEFGabcdefg])'
```
