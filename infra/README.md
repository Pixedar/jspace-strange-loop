# Unattended GPU runner

`job.py` runs a queue of commands (`/workspace/queue.txt`, appendable while running), uploads the output folder to a
Hugging Face dataset after every command, and exits when the queue stays empty. `run_v5.sh` wraps it with a deadline
(extendable), a watchdog that stops the instance an hour after the deadline, a final upload, and self-destruction of the
rented instance through the instance's own API key (`vast_self.sh`); no account key is ever placed on the instance.
