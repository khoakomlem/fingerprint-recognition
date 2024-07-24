import queue
import threading
import os


class ThreadPool:
    def __init__(self, num_threads=os.cpu_count(), verbose=True):
        self.verbose = verbose
        self.workers = []
        self.num_threads = num_threads
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.size = 0
        self.count = 0

        for _ in range(num_threads):
            worker = threading.Thread(target=self._worker)
            worker.daemon = True  # Threads will terminate when the main program ends
            worker.start()
            self.workers.append(worker)

    def _worker(self):
        while True:
            func, args, kwargs = self.tasks.get()
            try:
                result = func(*args, **kwargs)
                self.results.put(result)  # Put result into results queue
                self.count += 1
                percent = float("{:.1f}".format(self.count / self.size * 100))
                if self.verbose == True:
                    print(
                        f"========== Done ({self.count}/{self.size}) {percent}% =========="
                    )
            except Exception as e:
                print(f"Error executing function: {e}")
            finally:
                self.tasks.task_done()

    def reset(self, size):
        # ! MUST CALL THIS METHOD BEFORE ADDING NEW TASKS
        with self.tasks.mutex:
            self.tasks.queue.clear()
        with self.results.mutex:
            self.results.queue.clear()
        self.size = size
        self.count = 0

    def add_task(self, func, *args, **kwargs):
        self.tasks.put((func, args, kwargs))

    def wait_completion(self):
        self.tasks.join()
