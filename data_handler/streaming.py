import os
import logging
import fnmatch
import random
import numpy as np
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import sys
import traceback

"""
TensorFlow-free streaming utilities.
This file replaces previous tf.data/tf.io.gfile usage with standard
Python/Numpy implementations while keeping the same public APIs so that
train/test dataloaders continue to work unchanged.
"""

def get_files(dirname, filename_pat="*", recursive=False):
    if not os.path.exists(dirname):
        logging.warning(f"no file in {dirname} !")
        return []
    files = []
    for x in os.listdir(dirname):
        path = os.path.join(dirname, x)
        if os.path.isdir(path):
            if recursive:
                files.extend(get_files(path, filename_pat, recursive=True))
        elif fnmatch.fnmatch(x, filename_pat):
            files.append(path)
    return sorted(files)


def get_worker_files(dirnames,
                     worker_rank,
                     world_size,
                     filename_pat="*",
                     shuffle=False,
                     seed=0):
    """Get file paths belong to one worker."""
    all_files = []

    for dirname in dirnames:
        all_files.extend(get_files(dirname, filename_pat))

    files = []
    for i in range(worker_rank, len(all_files), world_size):
        files.append(all_files[i])

    if shuffle:
        random.shuffle(files)

    logging.info(
        f"worker_rank:{worker_rank}, world_size:{world_size}, shuffle:{shuffle}, seed:{seed}, directory:{dirname}, files:{files}"
    )
    return files


class StreamReader:
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        if isinstance(data_paths, (str, bytes)):
            data_paths = [data_paths]
        self.data_paths = list(data_paths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self._iter = None
        self._end = True

    def reset(self):
        paths = list(self.data_paths)
        if self.shuffle:
            random.shuffle(paths)

        def line_iter():
            for path in paths:
                try:
                    with open(path, 'rb') as f:  # return bytes, consistent with TF behavior
                        if self.shuffle:
                            # reservoir buffer for simple local shuffle per file
                            buf = []
                            for line in f:
                                buf.append(line.rstrip(b"\n"))
                                if len(buf) >= self.shuffle_buffer_size:
                                    random.shuffle(buf)
                                    for item in buf:
                                        yield item
                                    buf.clear()
                            if buf:
                                random.shuffle(buf)
                                for item in buf:
                                    yield item
                        else:
                            for line in f:
                                yield line.rstrip(b"\n")
                except FileNotFoundError:
                    logging.warning(f"file not found: {path}")
                    continue

        def batch_iter():
            batch = []
            for rec in line_iter():
                batch.append(rec)
                if len(batch) >= self.batch_size:
                    yield np.array(batch, dtype=object)
                    batch = []
            if batch:
                yield np.array(batch, dtype=object)

        self._iter = batch_iter()
        self._end = False

    def get_next(self):
        if self._iter is None:
            self.reset()
        try:
            return next(self._iter)
        except StopIteration:
            self._end = True
            return None

    def reach_end(self):
        return self._end


class StreamSampler:
    def __init__(
        self,
        data_dirs,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dirs,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.data_paths = data_paths
        self.stream_reader = StreamReader(data_paths, batch_size, enable_shuffle, shuffle_buffer_size)

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        """Implement iterator interface."""
        # logging.info(f"[StreamSampler] __next__")
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple) and not isinstance(next_batch, bytes):
            raise StopIteration
        # print(next_batch.shape)
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()


class StreamReaderForSpeedy:
    def __init__(self, file, batch_size):
        self.file = file
        self.stream_reader = StreamReader(file, batch_size)

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        """Implement iterator interface."""
        # logging.info(f"[StreamSampler] __next__")
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple) and not isinstance(next_batch, bytes):
            raise StopIteration
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()


class StreamSamplerTrainForSpeedyRec:
    def __init__(
        self,
        data_files,
        local_rank
    ):
        '''
        Args:
            data_files(manager.list()): the files storage train data
        '''
        files = []
        for i in range(local_rank, len(data_files), 8):
            files.append(data_files[i])
        self.data_files = files
        self.end = False
        self.sampler = None
        self.local_rank = local_rank

    def start_async(self):
        self.aval_count = 0
        self.end = False
        self.outputs = Queue(1000)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        try:
            self.sampler = self._generate_batch()
            for batch in self.sampler:
                if self.end:
                    break
                self.outputs.put(batch)
                self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _generate_batch(self):
        while True:
            if len(self.data_files) > 0:
                path = self.data_files.pop(0)
                market = os.path.basename(os.path.dirname(path))
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            yield line.strip('\n'), market
                except FileNotFoundError:
                    logging.warning(f"file not found: {path}")
                    continue
            else:
                self.end = True
                break

    def __iter__(self):
        self.join()
        self.start_async()
        return self

    def __next__(self):
        if self.sampler and  self.aval_count == 0 and self.end == True:
            raise StopIteration
        next_batch = self.outputs.get()
        self.outputs.task_done()
        self.aval_count -= 1
        return next_batch

    def join(self):
        self.end = True
        if self.sampler:
            while self.outputs.qsize() > 0:
                self.outputs.get()
                self.outputs.task_done()
            self.outputs.join()
            self.pool.shutdown(wait=True)
            logging.info("shut down pool.")
        self.sampler = None




class StreamReaderTest(StreamReader):
    def __init__(self, data_paths, batch_size, shuffle, shuffle_buffer_size=1000):
        super().__init__(data_paths, batch_size, shuffle=shuffle, shuffle_buffer_size=shuffle_buffer_size)


class StreamSamplerTest(StreamSampler):
    def __init__(
        self,
        data_dirs,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dirs,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.data_paths = data_paths
        self.stream_reader = StreamReaderTest(data_paths, batch_size, enable_shuffle, shuffle_buffer_size)
