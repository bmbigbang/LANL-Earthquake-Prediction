import pandas as pd
import io


class DataGen:
    # need generator-like reading of the file
    def __init__(self, file_path='train.csv', chunk_size=1000000):
        self.temp_handler = open(file_path, 'rb')
        self.col_names = [i.strip() for i in self.temp_handler.readline().decode("utf-8").split(",")]
        self.data_frame = pd.DataFrame()
        self.remaining = b''
        self.chunk_size = chunk_size

    def next_batch(self):
        temp_chars = self.read_in_chunks().__next__()

        if temp_chars:
            if self.remaining:
                temp_chars = self.remaining[::-1] + temp_chars
                self.remaining = b''
            while temp_chars and temp_chars[-1:] not in (b'\n', b'\r'):
                self.remaining += temp_chars[-1:]
                temp_chars = temp_chars[:-1]
        elif self.remaining:
            # this shouldnt ever happen
            # print('the remaining has been left with no content to add to')
            # temp_chars = self.remaining[::-1]
            # self.remaining = b''
            return pd.DataFrame(), True
        else:
            return pd.DataFrame(), True

        self.data_frame = pd.read_csv(io.BytesIO(temp_chars), encoding='utf8', header=None, names=self.col_names)

        return self.data_frame, False

    def read_in_chunks(self):
        while True:
            data = self.temp_handler.read(self.chunk_size)
            yield data

    def __exit__(self):
        self.temp_handler.close()
