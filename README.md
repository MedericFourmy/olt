Objects Localization and Tracking
=======================

Building dependencies 
---------------------
`conda create --name olt`  
`mamba env update --name olt -f environment.yaml`  

Then follow installation instructions for happypose (except for conda env creation)
- [happypose](https://github.com/agimus-project/happypose/tree/dev)

```bash
mamba install setproctitle
pip install thespian hypothesis pytest-benchmark pynvml
```

```bash
python olt/actor_tracker.py

# observe the some output
watch tail test.log
```

Killing the actors running in their processes is done by brute force (for now):
`kill -9 $(ps ax | grep Actor | fgrep -v grep | awk '{ print $1 }')`
