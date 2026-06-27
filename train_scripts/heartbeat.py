import time
from datetime import datetime

while True:
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    time.sleep(1)
